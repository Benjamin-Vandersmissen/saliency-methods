import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from base import SaliencyMethod
from utils import extract_layers

#
#  Learning Deep Features for Discriminative Localization (Zhou et al. 2016)
#


class _CAM(SaliencyMethod):

    def __init__(self, net: nn.Module, smoothed=False, smooth_rate=10):
        super().__init__(net, smoothed, smooth_rate)
        self.conv_layer: nn.Conv2d = None
        self.activation_hook = None
        self.conv_out = None
        self.layers = None

    def _find_conv_layer(self):
        """ Find the convolutional layer for calculating the activation maps."""
        for layer in self.layers[::-1]:
            if isinstance(layer, nn.Conv2d):
                self.conv_layer = layer
                break

    def _hook_conv_layer(self):
        """Hook the last convolutional layer to find its output activations."""

        def _conv_hook(module, inp, outp):
            self.conv_out = outp

        self.activation_hook = self.conv_layer.register_forward_hook(_conv_hook)

    def _get_weights(self, label: torch.Tensor) -> torch.Tensor:
        """Get the weights used for the CAM calculations

        Parameters
        ----------

        label : 1D-tensor that contains the label
            The label of which we want to calculate the weights.

        Returns
        -------

        weights : 3D-tensor of shape (channel, width, height)
            The weights used in the linear combination of activation maps.

        """
        raise NotImplementedError

    def _calculate(self, in_values: torch.tensor, label: torch.Tensor, resize: bool = True, **kwargs) -> np.ndarray:
        """ Calculate Class Activation Mapping for the given input.

        Parameters
        ----------

        in_values : 4D-tensor of shape (batch, channel, width, height)
            The image we want to explain. Only the first in the batch is considered.

        label : 1D-tensor
            The label we want to explain for.

        resize : boolean, default=True
            Resize the saliency map using bilinear interpolation.

        Returns
        -------

        3D-numpy.ndarray
            A saliency map for the first image in the batch.

        """

        weights = self._get_weights(label)

        saliency = torch.empty((3, self.conv_out.shape[2], self.conv_out.shape[3]))

        saliency[:] = F.relu((weights * self.conv_out.squeeze()).sum(dim=0))

        if resize:
            saliency = F.interpolate(saliency.unsqueeze(0), in_values.shape[2:], mode='bilinear').squeeze()

        saliency = saliency.detach().numpy()

        return saliency

    def calculate_map(self, in_values: torch.Tensor, label: torch.Tensor, resize: bool = True, **kwargs) -> np.ndarray:
        """ Calculate Class Activation Mapping for the given input.

        Parameters
        ----------

        in_values : 4D-tensor of shape (batch, channel, width, height)
            The image we want to explain. Only the first in the batch is considered.

        label : 1D-tensor
            The label we want to explain for.

        resize : boolean, default=True
            Resize the saliency map using bilinear interpolation.

        Returns
        -------

        3D-numpy.ndarray
            A saliency map for the first image in the batch.

        """
        self.layers = extract_layers(self.net, in_values.shape)
        if self.conv_layer is None:
            self._find_conv_layer()
            self._hook_conv_layer()

        _ = self.net(in_values)  # so we can find the hooked value

        return super().calculate_map(in_values, label, resize=resize, **kwargs)


class CAM(_CAM):

    def __init__(self, net: nn.Module, smoothed=False, smooth_rate=10):
        super().__init__(net, smoothed, smooth_rate)
        self.fc_layer: nn.Linear = None

    def _find_fc_layer(self):
        """ Find the linear layer for calculating the activation maps."""
        for layer in self.layers[::-1]:
            if isinstance(layer, nn.Linear):
                self.fc_layer = layer
                break

    def _get_weights(self, label: torch.Tensor) -> torch.Tensor:
        """ Get the weights used for the CAM calculations.

        Parameters
        ----------

        label : 1D-tensor that contains the label
            The label of which we want to calculate the weights.

        Returns
        -------

        weights : 3D-tensor of shape (channel, width, height)
            The weights used in the linear combination of activation maps.

        """
        weights = self.fc_layer.weight[label].squeeze()
        weights = weights.view((weights.shape[0], 1, 1))
        return weights

    def _calculate(self, in_values: torch.tensor, label: torch.Tensor, resize: bool = True, **kwargs) -> np.ndarray:
        """ Calculate Class Activation Mapping for the given input.

        Parameters
        ----------

        in_values : 4D-tensor of shape (batch, channel, width, height)
            The image we want to explain. Only the first in the batch is considered.

        label : 1D-tensor
            The label we want to explain for.

        resize : boolean, default=True
            Resize the saliency map using bilinear interpolation.

        Returns
        -------

        3D-numpy.ndarray
            A saliency map for the first image in the batch.

        """
        if self.fc_layer is None:
            self._find_fc_layer()

        return super()._calculate(in_values, label, resize, **kwargs)

    def calculate_map(self, in_values: torch.Tensor, label: torch.Tensor, resize: bool = True, **kwargs) -> np.ndarray:
        """ Calculate Class Activation Mapping for the given input.

        Parameters
        ----------

        in_values : 4D-tensor of shape (batch, channel, width, height)
            The image we want to explain. Only the first in the batch is considered.

        label : 1D-tensor
            The label we want to explain for.

        resize : boolean, default=True
            Resize the saliency map using bilinear interpolation.

        Returns
        -------

        3D-numpy.ndarray
            A saliency map for the first image in the batch.

        """
        return super().calculate_map(in_values, label, resize=resize, **kwargs)
