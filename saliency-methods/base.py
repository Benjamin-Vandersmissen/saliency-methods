import numpy as np
import torch
from torch import nn
from collections.abc import Callable
from abc import ABC, abstractmethod
import copy

#
#  A base class for the saliency methods
#
#  Smoothing is based on "SmoothGrad: removing noise by adding noise" (Smilkov et al. 2017)
#  https://arxiv.org/pdf/1706.03825.pdf
#


class SaliencyMethod(ABC):

    @staticmethod
    def gaussian_noise(in_values: torch.tensor, mean: float = 0, std: float = 0.1):
        """ Generate a noisy version of the input using gaussian noise.

        Parameters
        ----------
        in_values : torch.tensor
            The input value that will be transformed.

        mean : float, default=0
            The mean for the Gaussian distribution.

        std : float, default=0.1
            The standard deviation for the Gaussian distribution.

        Returns
        -------
        torch.Tensor
            The input with added Gaussian Noise.
        """

        return in_values + torch.normal(mean=mean, std=std, size=in_values.shape)

    def __init__(self, net: nn.Module, smoothed=False, smooth_rate=10, noise_function: Callable[[]]=gaussian_noise.__func__):
        """ Create a new SaliencyMethod object.

        Parameters
        ----------
        net : torch.nn.module
            The network to calculate saliency maps for.

        smoothed : bool, default=False
            Whether to apply SMOOTHGRAD or not.

        smooth_rate : int, default=10
            How many passes we use for calculating the smoothed saliency map.

        noise_function : callable, default=SaliencyMethod.gaussian_noise
            The noise function used in the SMOOTHGRAD algorithm.
        """

        self.smoothed = smoothed
        self.smooth_rate = smooth_rate
        self.noise_func = noise_function
        self.net = copy.deepcopy(net)

        self.net.eval()

    @abstractmethod
    def _calculate(self, in_values: torch.tensor, label: torch.Tensor, **kwargs) -> np.ndarray:
        """ Calculate a saliency map for the given input

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
        raise NotImplementedError("A Subclass of SaliencyMethod needs to implement this function")

    def calculate_map(self, in_values: torch.Tensor, label: torch.Tensor, **kwargs) -> np.ndarray:
        """ Calculates a saliency map for the given input and parameters. Smoothens the map if necessary.

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
        assert(in_values.shape[0] == 1 and len(in_values.shape) == 4)

        if self.smoothed:
            saliency_maps = np.empty((self.smooth_rate, *in_values.shape[1:]))
            for i in range(self.smooth_rate):
                noisy_input = self.noise_func(in_values.clone())
                saliency_maps[i] = self._calculate(noisy_input, label, **kwargs)
            saliency = saliency_maps.mean(axis=0).squeeze()
            return saliency
        else:
            return self._calculate(in_values, label, **kwargs)
