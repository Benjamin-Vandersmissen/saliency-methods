import numpy as np
import torch
from torch import nn
from collections.abc import Callable
from copy import deepcopy

from .base import SaliencyMethod
from .utils import extract_layers

__all__ = ['LRP']


class LRP(SaliencyMethod):
    #
    #  On Pixel-Wise Explanations for Non-Linear Classifier Decision by Layer-wise Relevance Propagation (Bach et al. 2015)
    #
    #  Implementation based on "Layerwise Relevance Propagation : an Overview” (Montavon et al. 2017)
    #

    def __init__(self, net: nn.Module, smoothed=False, smooth_rate=10):
        super().__init__(net, smoothed, smooth_rate)

        self.layers = None

    @staticmethod
    def _modify_layer(layer: nn.Module, func: Callable[[nn.Parameter], nn.Parameter]) -> nn.Module:
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

    # Remove nn,Flatten layer and replace nn.Linear layers with nn.Conv2D layers.
    def _process_layers(self, shape: tuple):
        """ Remove the flatten layer and replace the Linear layers with equivalent Conv2D layers.

        Parameters
        ----------

        shape : 4D-tuple of shape (batch, channel, width, height)
            The shape of the input the neural network expects.

        """
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

    # TODO: allow for different rules
    def _calculate(self, in_values: torch.tensor, label: torch.Tensor, **kwargs) -> np.ndarray:
        """ Calculates the Layer-wise Relevance Propagation w.r.t. the desired label.

        Parameters
        ----------

        in_values : 4D-tensor of shape (batch, channel, width, height)
            The image we want to explain. Only the first in the batch is considered.

        label : 1D-tensor
            The label we want to explain for.

        Returns
        -------

        3D-numpy.ndarray
            A saliency map for the first image in the batch.

        """
        if self.layers is None:
            self.layers = extract_layers(self.net, in_values.shape)
            self._process_layers(in_values.shape)

        activations = [in_values] + [None] * len(self.layers)
        for i in range(len(self.layers)):
            activations[i + 1] = self.layers[i].forward(activations[i])

        # Keep only the relevant score, mask the rest
        relevances = [torch.where(activations[-1] == activations[-1][0, label],
                                  activations[-1], torch.zeros_like(activations[-1]))]

        #  ! use .data to make a leaf node so we can use autograd
        for j in range(len(self.layers) - 1, 0, -1):
            activation = activations[j].data.requires_grad_(True)
            layer = self.layers[j]

            # As suggested in the paper, use 3 different rules, depending on the layer depth
            if j <= len(self.layers) // 3:
                rho = lambda p: p + 0.25 * p.clamp(min=0)
                incr = lambda z: z + 1e-9
            elif j <= (2 * len(self.layers)) // 3:
                rho = lambda p: p
                incr = lambda z: z + 1e-9 + 0.25 * ((z ** 2).mean() ** .5).data
            else:
                rho = lambda p: p
                incr = lambda z: z + 1e-9

            if isinstance(layer, nn.MaxPool2d):  # replace max pool with avg pool
                layer = nn.AvgPool2d(layer.kernel_size, layer.stride)

            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.MaxPool2d) or isinstance(layer, nn.AvgPool2d):
                z = incr(self._modify_layer(layer, rho).forward(activation))  # step 1
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
        z -= self._modify_layer(layer, lambda p: p.clamp(min=0)).forward(lb)  # - lb * w+
        z -= self._modify_layer(layer, lambda p: p.clamp(max=0)).forward(ub)  # - ub * w-

        s = (relevances[-1] / z).data  # step 2
        (z * s).sum().backward()  # step 3
        c, cp, cm = activation.grad, lb.grad, ub.grad
        relevances.append(activation * c + lb * cp + ub * cm)  # step 4

        saliency = relevances[-1].squeeze().detach().numpy()
        return saliency

    def calculate_map(self, in_values: torch.Tensor, label: torch.Tensor, **kwargs) -> np.ndarray:
        """ Calculates the Layer-wise Relevance Propagation w.r.t. the desired label. Smoothens the map if necessary

        Parameters
        ----------

        in_values : 4D-tensor of shape (batch, channel, width, height)
            The image we want to explain. Only the first in the batch is considered.

        label : 1D-tensor
            The label we want to explain for.

        Returns
        -------

        3D-numpy.ndarray
            A saliency map for the first image in the batch.
        """
        return super().calculate_map(in_values, label, **kwargs)
