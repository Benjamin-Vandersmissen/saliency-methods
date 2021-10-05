import numpy as np
import torch
from collections.abc import Callable
from .base import SaliencyMethod


class Smooth(SaliencyMethod):

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

        return in_values + torch.normal(mean=mean, std=std, size=in_values.shape).to(in_values.device)

    def __init__(self, method: SaliencyMethod, smooth_rate=10, noise_function = gaussian_noise.__func__):
        """ Create a new Smooth object.

        Parameters
        ----------
        method : SaliencyMethod
            The network to calculate saliency maps for.

        smooth_rate : int, default=10
            How many passes we use for calculating the smoothed saliency map.

        noise_function : callable, default=SaliencyMethod.gaussian_noise
            The noise function used in the smoothing algorithm.

        """
        super().__init__(method.net, method.device)

        self.smooth_rate = smooth_rate
        self.noise_func = noise_function
        self.method = method

    def calculate_map(self, in_values: torch.Tensor, label: torch.Tensor, **kwargs) -> np.ndarray:
        """ Smoothens a saliency map generated for a given input and label.

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
        in_values.to(self.device)

        saliency_maps = np.empty((self.smooth_rate, *in_values.shape[:]))
        for i in range(self.smooth_rate):
            noisy_input = self.noise_func(in_values.clone())
            saliency_maps[i, :] = self.method.calculate_map(noisy_input, label, **kwargs)
        saliency = saliency_maps.mean(axis=0).squeeze()
        return saliency
