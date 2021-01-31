from torch import nn
import torch
import numpy as np

from base import SaliencyMethod

#
#  Deep inside convolutional networks: Visualising image classification models and saliency maps. (Simonyan et al. 2013)
#  https://arxiv.org/pdf/1312.6034.pdf
#


class Gradient(SaliencyMethod):

    def __init__(self, net: nn.Module, smoothed=False, smooth_rate=10):
        super(Gradient, self).__init__(net, smoothed, smooth_rate)

    def _calculate(self, in_values: torch.Tensor, label: torch.Tensor, **kwargs) -> np.ndarray:
        """ Calculates the Gradient of the input w.r.t. the desired label.

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

        in_values = in_values.data.requires_grad_(True)
        self.net.zero_grad()

        out_values = self.net(in_values)[:, label]  # select relevant score
        out_values.backward()
        saliency = in_values.grad.squeeze().detach().numpy()

        return saliency

    def calculate_map(self, in_values: torch.Tensor, label: torch.Tensor, **kwargs) -> np.ndarray:
        """ Calculates the Gradient of the input w.r.t. the desired label. Smoothens the map if necessary.

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
