from abc import ABC, abstractmethod
from torch import nn
from collections.abc import Callable
from copy import deepcopy

from .utils import EPSILON

import torch


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

        try:
            new_layer.bias = nn.Parameter(func(layer.bias))
        except AttributeError:
            pass

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
    def __init__(self, mean=torch.zeros((1, 1, 1)), std=torch.ones((1, 1, 1))):
        super().__init__()
        self.mean = mean
        self.std = std

    def relevance(self, layer, activation, prev_relevance, layer_idx):
        device = activation.device
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)

        lb = (activation.data * 0 + (0-self.mean)/self.std).requires_grad_(True)  # lower bound on the activation
        ub = (activation.data * 0 + (1-self.mean)/self.std).requires_grad_(True)  # upper bound on the activation

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
