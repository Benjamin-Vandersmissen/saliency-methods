from torch import nn
import torch
import numpy as np

from .base import SaliencyMethod

__all__ = ['Gradient', 'GradientXInput', 'IntegratedGradient']


class Gradient(SaliencyMethod):
    """ Calculate the gradient of the input w.r.t. the desired label.

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

    Deep inside convolutional networks: Visualising image classification models and saliency maps. (Simonyan et al. 2013)
        https://arxiv.org/pdf/1312.6034.pdf
    """

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


class GradientXInput(Gradient):
    """ Calculate the gradient times the input w.r.t. the desired label.

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

    Learning Important Features Through Propagating Activation Differences (Shrikumar et al. 2017)
        https://arxiv.org/pdf/1312.6034.pdf
    """

    def __init__(self, net: nn.Module, smoothed=False, smooth_rate=10):
        super(Gradient, self).__init__(net, smoothed, smooth_rate)

    def _calculate(self, in_values: torch.Tensor, label: torch.Tensor, **kwargs) -> np.ndarray:
        """ Calculates the Gradient of the input w.r.t. the desired label and multiply it with the input.

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
        gradient = super()._calculate(in_values, label, **kwargs)
        saliency = gradient * in_values.squeeze().detach().numpy()

        return saliency


class IntegratedGradient(Gradient):
    """ Calculate the integrated gradient of the input w.r.t. the desired label.

    Parameters
    ----------

    net : torch.nn.module
        The network used for generating a saliency map.

    smoothed : bool
        Whether to apply smoothing via SMOOTHGRAD (True) or not (False).

    smooth_rate : int
        How many iterations of SMOOTHGRAD to use.

    baseline : str or 4D-tensor of shape (batch, channel, width, height)
        The baseline image to interpolate with the input. ("mean" = take the mean of the input image)

    nr_steps :
        How many steps of integrated gradients we use.

    References
    ----------

    Axiomatic Attribution for Deep Networks (Sundararajan et al. 2017)

    """

    def __init__(self, net: nn.Module, smoothed=False, smooth_rate=10, baseline="mean", nr_steps=50):
        super().__init__(net, smoothed, smooth_rate)
        self.baseline = baseline
        self.nr_steps = nr_steps

    def _calculate(self, in_values: torch.Tensor, label: torch.Tensor, **kwargs):
        """ Calculates the Integrated Gradient of the input w.r.t. the desired label.

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
        baseline = self.baseline
        if baseline is None:
            baseline = torch.zeros_like(in_values)
        elif baseline == "mean":
            baseline = torch.ones_like(in_values) * in_values.mean()
        else:
            assert(isinstance(baseline, torch.Tensor))
            assert(baseline.shape == in_values.shape)
        gradients = []

        for i in range(1, self.nr_steps + 1):
            current_input = baseline + (i / self.nr_steps) * (in_values - baseline)
            gradients.append(super(IntegratedGradient, self)._calculate(current_input, label, **kwargs))

        saliency = ((in_values - baseline) * np.average(gradients, axis=0)).squeeze().detach().numpy()
        return saliency
