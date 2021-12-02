import copy
from abc import ABC, abstractmethod
from typing import Any

from torch import nn
from collections.abc import Callable
from copy import deepcopy
from functools import partial

from .utils import EPSILON

import torch

# First for each module in the network that is supported (make list of supported modules) and check with in keyword:
# Rename the forward function to forward_orig and add the overridden forward module from LRPRule
# During the new forward pass, we call the original forward pass


class LRPRule(object):

    @staticmethod
    def relevance_func(inp, relevance, module):
        return inp

    def forward(self, input, relevance_func=relevance_func.__func__):
        return LRP_func.apply(self, input, relevance_func)


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
        module = Rule._modify_layer(module, rho)
        with torch.enable_grad():
            Z = incr(module.forward_orig(inp))
            Z = Z + (Z == 0) * EPSILON
            S = (relevance / Z).data
            c = torch.autograd.grad(Z, inp, S)[0]
            new_relevance = (inp * c).data
            print(module, new_relevance.sum())
            return new_relevance.data

    def forward(self, input):
        return LRPRule.forward(self, input, LRPZeroRule.relevance_func)


class LRPAlphaBetaRule(LRPRule):
    @staticmethod
    def relevance_func(inp, relevance, module, alpha=1, beta=0):
        with torch.enable_grad():
            pos_inp = torch.clip(inp, min=0).requires_grad_()
            neg_inp = torch.clip(inp, max=0).requires_grad_()

            pos_incr = lambda z: torch.clamp(z, min=0)
            neg_incr = lambda z: torch.clamp(z, max=0)

            pos_model = Rule._modify_layer(module, pos_incr)
            pos_model.bias = None
            neg_model = Rule._modify_layer(module, neg_incr)
            neg_model.bias = None

            zpos_pos = pos_model.forward_orig(pos_inp)
            zpos_neg = pos_model.forward_orig(neg_inp)
            zneg_neg = neg_model.forward_orig(neg_inp)
            zneg_pos = neg_model.forward_orig(pos_inp)

            Spos_pos = relevance / (zpos_pos + EPSILON)
            Spos_neg = relevance / (zpos_neg + EPSILON)
            Sneg_pos = relevance / (zneg_pos + EPSILON)
            Sneg_neg = relevance / (zneg_neg + EPSILON)

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
            z -= Rule._modify_layer(module, lambda p: p.clamp(min=0)).forward_orig(lb)  # - lb * w+
            z -= Rule._modify_layer(module, lambda p: p.clamp(max=0)).forward_orig(ub)  # - ub * w-

            z = z + (z == 0) * EPSILON  # for numerical stability

            s = (relevance / z).data  # step 2
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


class LRP_func(torch.autograd.Function):

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


class Rule(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def relevance(self, layer, activation, prev_relevance, layer_idx):
        raise NotImplementedError("A Subclass of Rule needs to implement this function")

    def __call__(self, layers, activations, labels):

        shape = activations[-1].shape[0:2]
        # Keep only the relevant score, mask the rest
        values = torch.gather(activations[-1].reshape(shape), 1, labels)

        relevances = [torch.zeros(shape).scatter_(1, labels, values).reshape(*shape, 1, 1)]

        #  ! use .data to make a leaf node so we can use autograd
        for j in range(len(layers) - 1, -1, -1):
            activation = activations[j].data.requires_grad_(True)
            layer = layers[j]

            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.MaxPool2d) or isinstance(layer, nn.AvgPool2d):
                relevances.append(self.relevance(layer, activation, relevances[-1], j))

            else:
                relevances.append(relevances[-1])

        return relevances[-1].detach().cpu().numpy()

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


class LRP0(Rule):
    def __init__(self):
        super().__init__()
        self.rho = lambda p: p
        self.incr = lambda z: z

    def relevance(self, layer, activation, prev_relevance, layer_idx):
        z = self.incr(self._modify_layer(layer, self.rho).forward(activation))  # step 1
        z[z == 0] = EPSILON  # for numeric stability
        s = (prev_relevance / z).data  # step 2
        (z * s).sum().backward()  # step 3
        c = activation.grad
        return (activation * c).data


class LRPeps(LRP0):
    def __init__(self, eps="auto"):
        super().__init__()
        if eps == "auto":
            self.incr = lambda z: z + 0.25 * ((z ** 2).mean() ** .5).data
        else:
            self.incr = lambda z: z + eps


class LRPgamma(LRP0):
    def __init__(self, gamma=0.25):
        super().__init__()
        self.rho = lambda p: p + gamma * p.clamp(min=0)


class ZbRule(Rule):
    def __init__(self, mean=torch.zeros((1, 3, 1, 1)), std=torch.ones((1, 3, 1, 1))):
        super().__init__()
        self.mean = mean
        self.std = std

    def relevance(self, layer, activation, prev_relevance, layer_idx):
        device = activation.device
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)

        lb = (activation.data * 0 - self.std).requires_grad_(True)  # lower bound on the activation
        ub = (activation.data * 0 + self.std).requires_grad_(True)  # upper bound on the activation

        z = layer.forward(activation)
        z -= self._modify_layer(layer, lambda p: p.clamp(min=0)).forward(lb)  # - lb * w+
        z -= self._modify_layer(layer, lambda p: p.clamp(max=0)).forward(ub)  # - ub * w-

        z[z == 0] = EPSILON  # for numerical stability

        s = (prev_relevance / z).data  # step 2
        (z * s).sum().backward()  # step 3
        c, cp, cm = activation.grad, lb.grad, ub.grad
        return activation * c + lb * cp + ub * cm  # step 4


# combine multiple rules.
class CompositeRule(Rule):

    def __init__(self, rules=None, pixel=None, feature=None):
        super().__init__()

        self.rules = []

        if isinstance(rules, dict):
            keys = sorted([key for key in rules.keys()])
            self.rules = []
            prev_idx = 0
            for key in keys:
                rule = rules[key]
                if len(self.rules) == 0:
                    self.rules += [rule]
                else:
                    self.rules += (key-prev_idx-1)*[self.rules[-1]]
                    self.rules += [rule]
                prev_idx = key

            #TODO: assert on structure of rules : only numeric keys, or list of features (should match).
        elif rules is None and pixel is not None and feature is not None:
            self.rules = {"pixel": pixel, "feature": feature}

        pass

    def relevance(self, layer, activation, prev_relevance, layer_idx):
        if "pixel" in self.rules and layer_idx == 0:
            return self.rules["pixel"].relevance(layer, activation, prev_relevance, layer_idx)

        elif "feature" in self.rules and layer_idx != 0:
            return self.rules["feature"].relevance(layer, activation, prev_relevance, layer_idx)

        else:
            return self.rules[min(layer_idx, len(self.rules)-1)].relevance(layer, activation, prev_relevance, layer_idx) # Failsafe for if there are more layers than specified
