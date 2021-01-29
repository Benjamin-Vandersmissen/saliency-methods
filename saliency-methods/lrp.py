import numpy as np
import torch
from torch import nn
from collections.abc import Callable
from copy import deepcopy

from base import SaliencyMethod


class LRP(SaliencyMethod):
    def __init__(self, net: nn.Module,smoothed=False, smooth_rate=10):
        super().__init__(net, smoothed, smooth_rate)

        self.layers = None

    @staticmethod
    def __newlayer(layer: nn.Module, func: Callable[[nn.Parameter], nn.Parameter]) -> nn.Module:
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

    # Extract the layers used in the network in the order they are applied.
    @staticmethod
    def __extract_layers(net: nn.Module, shape: tuple) -> list:
        dummy_value = torch.zeros(shape)
        layers, handles = [], []

        func = lambda module, input, output: layers.append(module)

        for name, m in net.named_modules():
            if len(list(m.named_modules())) == 1:  # Only ever have the singular layers, no sub networks etc
                handles.append(m.register_forward_hook(func))

        net.forward(dummy_value)
        for handle in handles:
            handle.remove()

        return layers

    # Remove nn,Flatten layer and replace nn.Linear layers with nn.Conv2D layers
    def __process_layers(self, shape: tuple):
        dummy_value = torch.zeros(shape)

        new_layers = []
        for layer in self.layers:

            if isinstance(layer, nn.Linear):  # convert linear layers to corresponding convolutional layers
                new_layer = None
                if dummy_value.shape[2] != 1:  # This layer replaces the flatten layer and the first linear layer
                    m, n = dummy_value.shape[1], layer.weight.shape[0]
                    new_layer = nn.Conv2d(m, n, (dummy_value.shape[2], dummy_value.shape[3]))  # Flatten into 1x1 output
                    new_layer.weight = nn.Parameter(layer.weight.reshape(n, m, dummy_value.shape[2], dummy_value.shape[3]))
                else:
                    m, n = layer.weight.shape[1], layer.weight.shape[0]
                    new_layer = nn.Conv2d(m, n, 1)
                    new_layer.weight = nn.Parameter(layer.weight.reshape(n, m, 1, 1))
                new_layer.bias = nn.Parameter(layer.bias)
                new_layers.append(new_layer)
                dummy_value = new_layer.forward(dummy_value)

            elif not isinstance(layer, nn.Flatten):
                dummy_value = layer.forward(dummy_value)
                new_layers.append(layer)

        self.layers = new_layers

    def calculate_mask(self, in_values: torch.Tensor, label: torch.Tensor) -> np.ndarray:
        if self.layers is None:
            self.layers = self.__extract_layers(self.net, in_values.shape)
            self.__process_layers(in_values.shape)

        activations = [in_values] + [None] * len(self.layers)
        for i in range(len(self.layers)):
            activations[i + 1] = self.layers[i].forward(activations[i])

        # Keep only the relevant score, mask the rest
        relevances = [torch.where(activations[-1] == activations[-1][0, label],
                                  activations[-1], torch.zeros_like(activations[-1]))]

        #  ! use .data to make a leaf node so we can use autograd
        for j in range(len(self.layers) - 1, 0, -1):
            activation = (activations[j].data).requires_grad_(True)
            layer = self.layers[j]

            rho = lambda p: p  # Default rules.
            incr = lambda z: z + 1e-9  # Default rules.

            if isinstance(layer, nn.MaxPool2d):  # replace max pool with avg pool
                layer = nn.AvgPool2d(layer.kernel_size, layer.stride)

            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.MaxPool2d) or isinstance(layer, nn.AvgPool2d):
                z = incr(self.__newlayer(layer, rho).forward(activation))  # step 1
                s = (relevances[-1] / z).data  # step 2
                (z * s).sum().backward()  # step 3
                c = activation.grad
                relevances.append((activation * c).data)

            else:
                relevances.append(relevances[-1])

        # pixel level uses the zb rule from Taylor Decomposition
        activation = activations[0].data.requires_grad_(True)  # this is the pixel level
        layer = self.layers[0]

        # TODO: these need to be configured by the user
        lb = (activation.data * 0).requires_grad_(True)  # lower bound on the activation
        ub = (activation.data * 0 + 1).requires_grad_(True)  # upper bound on the activation

        z = layer.forward(activation) + 1e-9
        z -= self.__newlayer(layer, lambda p: p.clamp(min=0)).forward(lb)  # - lb * w+
        z -= self.__newlayer(layer, lambda p: p.clamp(max=0)).forward(ub)  # - ub * w-

        s = (relevances[-1] / z).data  # step 2
        (z * s).sum().backward()  # step 3
        c, cp, cm = activation.grad, lb.grad, ub.grad
        relevances.append(activation * c + lb * cp + ub * cm)  # step 4

        saliency = relevances[-1].squeeze().detach().numpy()
        return saliency
