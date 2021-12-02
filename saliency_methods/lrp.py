import copy
import types

import numpy as np
import torch
import torch.nn.functional as F

from .base import SaliencyMethod
from .utils import extract_layers
from .rule import *

__all__ = ['LRP']

# TODO: https://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_custom_function.html
# TODO: https://github.com/dmitrysarov/LRP_decomposition/tree/72031a5252aafb7f7e51abe28bba40dba0fce9e7
# Implement rules as custom autograd functions, with forward and backward.


class LRPBackward(SaliencyMethod):
    def __init__(self, net: nn.Module, **kwargs):
        super().__init__(net, **kwargs)

        first_layer = True
        for module in self.net.modules():
            if len(list(module.modules())) == 1:
                setattr(module, 'forward_orig', module.forward)
                if not isinstance(module, nn.ReLU) or isinstance(module, nn.BatchNorm2d):
                    if first_layer:
                        setattr(module, 'forward', types.MethodType(getattr(LRPZbRule, 'forward'), module))
                        first_layer = False
                    else:
                        setattr(module, 'forward', types.MethodType(getattr(LRPAlphaBetaRule, 'forward'), module))

                else:
                    setattr(module, 'forward', types.MethodType(getattr(LRPIdentityRule, 'forward'), module))

    def calculate_map(self, in_values: torch.tensor, labels: torch.Tensor, **kwargs) -> np.ndarray:
        in_values.to(self.device).requires_grad_(True)
        labels = labels.reshape(labels.shape[0], 1)
        out_values = torch.gather(self.net(in_values), 1, labels)  # select relevant score
        out_values.backward(gradient=torch.ones_like(out_values))
        return in_values.grad.detach().numpy()


class LRP(SaliencyMethod):
    """
    On Pixel-Wise Explanations for Non-Linear Classifier Decision by Layer-wise Relevance Propagation (Bach et al. 2015)
    """

    def __init__(self, net: nn.Module, rule: Rule = CompositeRule({0:ZbRule(), 1:LRPeps()}), **kwargs):
        """
        Initialize a new LRP Saliency Method object.
        :param net: The neural network to use.
        :param rule: The rule(s) to apply during the propagation process.
        :param kwargs: Other arguments.
        """
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
        for _, layer in self.layers:

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
        """ Calculates the Layer-wise Relevance Propagation of the input w.r.t. the desired label.

        Parameters
        ----------

        in_values : 4D-tensor of shape (batch, channel, width, height)
            A batch of images we want to generate saliency maps for.

        labels : 1D-tensor containing *batch* elements.
            The labels for the images we want to explain for.

        Returns
        -------

        4D-numpy.ndarray
            A batch of saliency maps for the images and labels provided.

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


class MarginalWinningProbability(SaliencyMethod):
    def __init__(self, net : nn.Module, **kwargs):
        super(MarginalWinningProbability, self).__init__(net)

        self.activations = []
        self.maps = []

        self.forward_hooks = []
        self.backward_hooks = []

    def calculate_map(self, in_values: torch.tensor, labels: torch.Tensor, **kwargs) -> np.ndarray:
        pass
