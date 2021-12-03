import types

import numpy as np
import torch
import torch.nn as nn

from .base import SaliencyMethod
from .rule import *

__all__ = ['LRP']


class LRP(SaliencyMethod):
    """
    On Pixel-Wise Explanations for Non-Linear Classifier Decision by Layer-wise Relevance Propagation (Bach et al. 2015)
    Special thanks to:
    https://github.com/wjNam/Relative_Attributing_Propagation for an LRP implementation working with Resnet to compare against
    https://github.com/dmitrysarov/LRP_decomposition for a working LRP implementation using autograd.
    """
    def __init__(self, net: nn.Module, **kwargs):
        """
        Initialize a new LRP Saliency Method object.
        :param net: The neural network to use.
        :param kwargs: Other arguments.
        """
        super().__init__(net, **kwargs)

        self.assign_rules()

    def calculate_map(self, in_values: torch.tensor, labels: torch.Tensor, **kwargs) -> np.ndarray:
        """ Calculates the LRP map of the input w.r.t. the desired labels.

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

        in_values.to(self.device).requires_grad_(True)
        labels = labels.reshape(labels.shape[0], 1)
        out_values = self.net(in_values)
        # All 0's except the original values on the positions of the label.
        grad_out = torch.scatter(torch.zeros_like(out_values), 1, labels, torch.gather(out_values, 1, labels))
        grad = torch.autograd.grad(out_values, in_values, grad_out)[0]
        return grad.detach().numpy()

    def assign_rules(self):
        """ Assign LRP rules to each layer in the neural network

        Throws
        ------
        Exception if a layer has no associated rule.
        """
        identity_layers = [nn.ReLU, nn.BatchNorm2d, nn.Dropout]
        zero_layers = [nn.AdaptiveAvgPool2d, nn.AvgPool2d, nn.MaxPool2d]
        alpha_beta_layers = [nn.Conv2d, nn.Linear]

        layer_map = {layer: LRPIdentityRule for layer in identity_layers}
        layer_map.update({layer: LRPZeroRule for layer in zero_layers})
        layer_map.update({layer: LRPAlphaBetaRule for layer in alpha_beta_layers})

        initial_layer = True
        for layer in self.net.modules():
            if len(list(layer.children())) == 0:
                if layer.__class__ not in layer_map:
                    raise Exception("There is no rule associated with this layer! (%s)" % layer.__class__)

                setattr(layer, 'forward_orig', layer.forward)
                if initial_layer:
                    setattr(layer, 'forward', types.MethodType(getattr(LRPZbRule, 'forward'), layer))
                    initial_layer = False
                else:
                    setattr(layer, 'forward', types.MethodType(getattr(layer_map[layer.__class__], 'forward'), layer))


class MarginalWinningProbability(SaliencyMethod):
    def __init__(self, net : nn.Module, **kwargs):
        super(MarginalWinningProbability, self).__init__(net)

        self.activations = []
        self.maps = []

        self.forward_hooks = []
        self.backward_hooks = []

    def calculate_map(self, in_values: torch.tensor, labels: torch.Tensor, **kwargs) -> np.ndarray:
        pass
