import numpy as np
import torch
from torch import nn
from collections.abc import Callable
from copy import deepcopy

from .base import SaliencyMethod
from .utils import extract_layers
from .rule import *

__all__ = ['LRP']


class LRP(SaliencyMethod):
    """ Calculate GradCAM for the input w.r.t. the desired label.

    Parameters
    ----------

    net : torch.nn.module
        The network used for generating a saliency map.

    smoothed : bool
        Whether to apply smoothing via SMOOTHGRAD (True) or not (False).

    smooth_rate : int
        How many iterations of SMOOTHGRAD to use.

    References
    ----------

    On Pixel-Wise Explanations for Non-Linear Classifier Decision by Layer-wise Relevance Propagation (Bach et al. 2015)
    """

    def __init__(self, net: nn.Module, rule: Rule = CompositeRule({0:ZbRule(), 1:LRP0()}), **kwargs):
        super().__init__(net, **kwargs)

        self.layers = None
        self.rule = rule

    def _process_layers(self, shape: tuple):
        """ Remove the flatten layer and replace the Linear layers with equivalent Conv2D layers.

        Parameters
        ----------

        shape : 4D-tuple of shape (batch, channel, width, height)
            The shape of the input the neural network expects.

        """
        dummy_value = torch.zeros(shape).to(self.device)

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
                new_layers.append(new_layer.to(self.device))
                dummy_value = new_layer.forward(dummy_value)

            elif not isinstance(layer, nn.Flatten):
                dummy_value = layer.forward(dummy_value)

                if isinstance(layer, nn.MaxPool2d):
                    new_layers.append(nn.AvgPool2d(layer.kernel_size, layer.stride))
                else:
                    new_layers.append(layer)

        self.layers = new_layers

    def calculate_map(self, in_values: torch.tensor, labels: torch.Tensor, **kwargs) -> np.ndarray:
        """ Calculates the Layer-wise Relevance Propagation w.r.t. the desired label.

        Parameters
        ----------

        in_values : 4D-tensor of shape (batch, channel, width, height)
            The batch of images we want to explain.

        labels : 1D-tensor of *batch* elements
            The labels we want to explain for.

        Returns
        -------

        4D-numpy.ndarray
            A saliency map for the first image in the batch.

        """
        in_values.to(self.device)
        labels = labels.reshape(labels.shape[0], 1)

        if self.layers is None:
            self.layers = extract_layers(self.net, in_values.shape)
            self._process_layers(in_values.shape)

        activations = [in_values] + [None] * len(self.layers)
        for i in range(len(self.layers)):
            activations[i + 1] = self.layers[i].forward(activations[i])

        return self.rule(self.layers, activations, labels)

