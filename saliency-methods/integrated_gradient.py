import numpy as np
import torch
from torch import nn

from gradient import Gradient

#
#  Axiomatic Attribution for Deep Networks (Sundararajan et al. 2017)
#


class IntegratedGradient(Gradient):

    def __init__(self, net: nn.Module, smoothed=False, smooth_rate=10):
        super().__init__(net, smoothed, smooth_rate)

    def _calculate(self, in_values: torch.Tensor, label: torch.Tensor, baseline: torch.Tensor = None,
                   nr_steps: int = 50, **kwargs):
        """ Calculates the Integrated Gradient of the input w.r.t. the desired label.

        Parameters
        ----------

        in_values : 4D-tensor of shape (batch, channel, width, height)
            The image we want to explain. Only the first in the batch is considered.

        label : 1D-tensor
            The label we want to explain for.

        baseline : 4D-tensor of shape (batch, channel, width, height), default : The all-zero image
            The baseline image used for the interpolation.

        nr_steps : int, default=50
            The amount of interpolation steps used.

        Returns
        -------

        3D-numpy.ndarray
            A saliency map for the first image in the batch.

        """
        if baseline is None:
            baseline = torch.zeros_like(in_values)

        gradients = []

        for i in range(1, nr_steps + 1):
            current_input = baseline + (i / nr_steps) * (in_values - baseline)
            gradients.append(super(IntegratedGradient, self)._calculate(current_input, label, **kwargs))

        saliency = ((in_values - baseline) * np.average(gradients, axis=0)).squeeze().detach().numpy()
        return saliency

    def calculate_map(self, in_values: torch.Tensor, label: torch.Tensor, baseline: torch.Tensor = None,
                      nr_steps: int = 50, **kwargs) -> np.ndarray:

        """ Calculates the Integrated Gradient of the input w.r.t. the desired label. Smoothens the map if necessary.

        Parameters
        ----------

        in_values : 4D-tensor of shape (batch, channel, width, height)
            The image we want to explain. Only the first in the batch is considered.

        label : 1D-tensor
            The label we want to explain for.

        baseline : 4D-tensor of shape (batch, channel, width, height), default : The all-zero image
            The baseline image used for the interpolation.

        nr_steps : int, default=50
            The amount of interpolation steps used.

        Returns
        -------

        3D-numpy.ndarray
            A saliency map for the first image in the batch.

        """

        return super().calculate_map(in_values, label, baseline=baseline, nr_steps=nr_steps)
