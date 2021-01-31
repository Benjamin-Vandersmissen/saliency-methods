import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from base import SaliencyMethod
from utils import extract_layers

#
#  Learning Deep Features for Discriminative Localization (Zhou et al. 2016)
#


class CAM(SaliencyMethod):

    def __init__(self, net: nn.Module, smoothed=False, smooth_rate=10):
        super().__init__(net, smoothed, smooth_rate)
        self.conv_layer: nn.Conv2d = None
        self.conv_hook = None
        self.conv_out = None
        self.fc_layer: nn.Linear = None

    def _find_conv_fc_layer(self, shape: tuple):
        """ Find the Linear (Fully Connected) layer and the last Convolutional layer.

        Parameters
        ----------

        shape : 4D-tuple of shape (batch, channel, width, height)
            The shape of the input for the network.

        """
        layers = extract_layers(self.net, shape)
        for layer in layers[::-1]:
            if self.fc_layer is None:
                if isinstance(layer, nn.Linear):
                    self.fc_layer = layer
                elif isinstance(layer, nn.Conv2d):
                    raise Exception("The last convolutional layer must directly feed into the GAP layer "
                                    "and a singular FC layer")
            else:
                if isinstance(layer, nn.Linear):
                    raise Exception("There may only be one FC layer in the network")
                elif isinstance(layer, nn.Conv2d):
                    self.conv_layer = layer
                    break

        if self.conv_layer.out_channels != self.fc_layer.in_features:
            raise Exception("Number of outgoing channels of the last convolutional layer doesn't match the number of "
                            "incoming features in the linear layer ({} vs {})\n Are you using a GAP layer?"
                            .format(self.conv_layer.out_channels, self.fc_layer.in_features))

    def _hook_conv_layer(self):
        """Hook the last convolutional layer to find its output activations."""

        def _conv_hook(module, inp, outp):
            self.conv_out = outp

        self.conv_hook = self.conv_layer.register_forward_hook(_conv_hook)

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
        if self.conv_layer is None:
            self._find_conv_fc_layer(in_values.shape)
            self._hook_conv_layer()

        _ = self.net(in_values)  # so we can find the hooked value

        weight = self.fc_layer.weight[label].squeeze()
        weight = weight.view((weight.shape[0], 1, 1))

        saliency = torch.empty((3, self.conv_out.shape[2], self.conv_out.shape[3]))

        saliency[:] = F.relu((weight * self.conv_out.squeeze()).sum(dim=0))

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
        return super().calculate_map(in_values, label, resize=resize, **kwargs)
