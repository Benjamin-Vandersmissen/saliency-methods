import copy
from typing import Any

from torch import nn
from copy import deepcopy
from functools import partial

from .utils import safe_divide

import torch

__all__ = ['LRPfunc', 'LRPRule', 'LRPZeroRule', 'LRPIdentityRule', 'LRPEpsilonRule', 'LRPgammaRule', 'LRPAlphaBetaRule',
           'LRPZbRule']

# The following classes implement different Rules for Layerwise-Relevance Propagation.
# The LRP process is implemented efficiently using the pytorch autograd mechanism. To accomplish this, we introduced a
# new class LRPfunc, which is a subclass of an autograd function and enables a module to backpropagate relevance.

# Each rule has a forward method associated and to implement a rule for an instance of a module, the module::forward()
# function needs to be renamed to module::forward_orig(),
# while a new bound method Rule::forward() is bound to module::forward().
# This can be done using '''setattr(module, 'forward_orig', module.forward)''' and
# '''setattr(module, 'forward', types.MethodType(getattr(Rule, 'forward'), module))
#
# Rule::forward() calls the overridden LRPfunc::forward() and provides it with the original input,
# module and the chosen relevance function.
# LRPfunc::forward() then calls module::forward_orig() on the input and saves the module, input and relevance function
# During Rule::backward, LRPfunc::backward is automatically called due to the autograd mechanism.
# LRPfunc::backward invokes the relevance function with the needed arguments and backpropagates the results.


#
# The following rules are implemented:
# - LRPRule : returns the gradient
# - LRPZeroRule : use LRP-0
# - LRPIdentityRule : returns the relevance of the previous layer
# - LRPEpsilonRule : use LRP-\eps (denoise the relevance from contradictory / weak contributions)
# - LRPGammaRule : use LRP-\gamma (add more weight to positive contributions)
# - LRPAlphaBetaRule : use LRP-\alpha,\beta (adds weights for positive and negative contributions
# - LRPZbRule : implements the Zb rule (specific rule for the input layer.)

class LRPRule(object):

    @staticmethod
    def relevance_func(inp, relevance, module):
        return inp


    @staticmethod
    def _modify_layer(layer: nn.Module, func) -> nn.Module:
        """ Modify a layer by applying a function on the weights and biases.

        Parameters
        ----------

        layer : torch.nn.module
            The layer to modify.

        func : callable
            The function that will modify the weights and biases.

        Returns
        -------

        new_layer : torch.nn.module
            The layer with modified weights and biases.

        """
        new_layer = deepcopy(layer)

        try:
            new_layer.weight = nn.Parameter(func(layer.weight))
        except AttributeError:
            pass

        # try:
        #     if layer.bias is not None:
        #         new_layer.bias = nn.Parameter(func(layer.bias))
        # except AttributeError:
        #     pass

        return new_layer

    def forward(self, input, relevance_func=relevance_func.__func__):
        return LRPfunc.apply(self, input, relevance_func)


class LRPIdentityRule(LRPRule):

    @staticmethod
    def relevance_func(inp, relevance, module):
        return relevance

    def forward(self, input):
        return LRPRule.forward(self, input, LRPIdentityRule.relevance_func)


class LRPZeroRule(LRPRule):

    @staticmethod
    def relevance_func(inp, relevance, module, incr=lambda z: z, rho=lambda p: p):
        inp.requires_grad_(True)
        if inp.grad is not None:
            inp.grad.zero_()
        module = LRPRule._modify_layer(module, rho)
        with torch.enable_grad():
            Z = incr(module.forward_orig(inp))
            S = safe_divide(relevance, Z).data
            c = torch.autograd.grad(Z, inp, S)[0]
            new_relevance = (inp * c).data
            return new_relevance.data

    def forward(self, input):
        return LRPRule.forward(self, input, LRPZeroRule.relevance_func)


class LRPAlphaBetaRule(LRPRule):
    @staticmethod
    def relevance_func(inp, relevance, module, alpha=1, beta=0):
        with torch.enable_grad():
            pos_inp = torch.clip(inp, min=0).requires_grad_(True)
            neg_inp = torch.clip(inp, max=0).requires_grad_(True)

            pos_incr = lambda z: torch.clamp(z, min=0)
            neg_incr = lambda z: torch.clamp(z, max=0)

            pos_model = LRPRule._modify_layer(module, pos_incr)
            pos_model.bias = None
            neg_model = LRPRule._modify_layer(module, neg_incr)
            neg_model.bias = None

            zpos_pos = pos_model.forward_orig(pos_inp)
            zpos_neg = pos_model.forward_orig(neg_inp)
            zneg_neg = neg_model.forward_orig(neg_inp)
            zneg_pos = neg_model.forward_orig(pos_inp)

            Spos_pos = safe_divide(relevance, zpos_pos)
            Spos_neg = safe_divide(relevance, zpos_neg)
            Sneg_pos = safe_divide(relevance, zneg_pos)
            Sneg_neg = safe_divide(relevance, zneg_neg)

            Cpos_pos = pos_inp * torch.autograd.grad(zpos_pos, pos_inp, Spos_pos)[0]
            Cpos_neg = neg_inp * torch.autograd.grad(zpos_neg, neg_inp, Spos_neg)[0]
            Cneg_pos = pos_inp * torch.autograd.grad(zneg_pos, pos_inp, Sneg_pos)[0]
            Cneg_neg = neg_inp * torch.autograd.grad(zneg_neg, neg_inp, Sneg_neg)[0]

            activator_relevance = Cpos_pos + Cneg_neg
            inhibitor_relevance = Cneg_pos + Cpos_neg

            return alpha * activator_relevance + beta * inhibitor_relevance

    def forward(self, input):
        return LRPRule.forward(self, input, LRPAlphaBetaRule.relevance_func)


class LRPEpsilonRule(LRPZeroRule):
    def forward(self, input):
        incr = lambda z: z + 0.25 * ((z ** 2).mean() ** .5).data
        return LRPRule.forward(self, input, partial(LRPZeroRule.relevance_func, incr=incr))


class LRPZbRule(LRPRule):

    @staticmethod
    def relevance_func(inp, relevance, module, lower=-torch.ones((1,3,1,1)), upper=torch.ones((1, 3, 1, 1))):
        with torch.enable_grad():
            inp.requires_grad_(True)

            lb = (inp * 0 + lower).requires_grad_(True)  # lower bound on the activation
            ub = (inp * 0 + upper).requires_grad_(True)  # upper bound on the activation

            lb.retain_grad()
            ub.retain_grad()

            z = module.forward_orig(inp)
            z -= LRPRule._modify_layer(module, lambda p: p.clamp(min=0)).forward_orig(lb)  # - lb * w+
            z -= LRPRule._modify_layer(module, lambda p: p.clamp(max=0)).forward_orig(ub)  # - ub * w-

            s = safe_divide(relevance, z).data  # step 2
            (z * s).sum().backward()  # step 3
            c, cp, cm = inp.grad, lb.grad, ub.grad
            return inp * c + lb * cp + ub * cm  # step 4

    def forward(self, input, lower=-torch.ones((1,3,1,1)), upper=torch.ones((1,3,1,1))):
        return LRPRule.forward(self, input, partial(LRPZbRule.relevance_func, lower=lower, upper=upper))


class LRPgammaRule(LRPZeroRule):
    def forward(self, input, gamma=0.25):
        rho = lambda p: p + gamma * p.clamp(min=0)
        return LRPRule.forward(self,input, partial(LRPZeroRule.relevance_func, rho=rho))

# class LRPbatchnormRule(LRPRule):
#
#     @staticmethod
#     def relevance_func(inp, relevance, module):
#         outp = module(inp)
#         bias = module.bias.view((1, -1, 1, 1))
#         running_mean = module.running_mean.view((1, -1, 1, 1))
#         new_relevance = inp*(outp - bias)*relevance/((inp - running_mean)*outp + EPSILON)
#         print(module, relevance.sum().item())
#         return new_relevance
#
#     def forward(self, input):
#         return LRPRule.forward(self, input, LRPbatchnormRule.relevance_func)


class LRPfunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx: Any, module, inp, relevance_func) -> Any:
        ctx.relevance_func = relevance_func
        ctx.inp = inp.data
        ctx.module = copy.deepcopy(module)
        return module.forward_orig(inp)

    @staticmethod
    def backward(ctx: Any, relevance) -> Any:
        new_relevance = ctx.relevance_func(ctx.inp, relevance, ctx.module)
        return None, new_relevance, None
