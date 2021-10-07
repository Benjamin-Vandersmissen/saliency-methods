import numpy as np
import torch
from collections.abc import Callable
from .base import SaliencyMethod


class Smooth(SaliencyMethod):
    """
    Based on SmoothGrad: removing noise by adding noise (Smilkov et al. 2017)
    """
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
        """
        Initialize a new Smooth object.
        :param method: The saliency method to apply smoothing to.
        :param smooth_rate: How many noisy inputs to generate.
        :param noise_function: The function to generate noisy input values.
        """
        super().__init__(method.net, method.device)

        self.smooth_rate = smooth_rate
        self.noise_func = noise_function
        self.method = method

    def calculate_map(self, in_values: torch.Tensor, labels: torch.Tensor, **kwargs) -> np.ndarray:
        """ Smoothens a saliency method of the input w.r.t. the desired label.

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

        saliency_maps = np.empty((self.smooth_rate, *in_values.shape[:]))
        for i in range(self.smooth_rate):
            noisy_input = self.noise_func(in_values.clone())
            saliency_maps[i, :] = self.method.calculate_map(noisy_input, labels, **kwargs)
        saliency = saliency_maps.mean(axis=0).squeeze()
        return saliency
