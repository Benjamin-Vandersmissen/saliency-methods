import numpy as np
import torch
from .base import CompositeSaliencyMethod, SaliencyMethod
from .gradient import GuidedBackProp

__all__ = ["Smooth", "Var", "Guided"]


class Smooth(CompositeSaliencyMethod):
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

    def __init__(self, method: SaliencyMethod, smooth_rate=10, noise_function=gaussian_noise.__func__):
        """
        Initialize a new Smooth object.
        :param method: The saliency method to apply smoothing to.
        :param smooth_rate: How many noisy inputs to generate.
        :param noise_function: The function to generate noisy input values.
        """
        super().__init__(method)

        self.smooth_rate = smooth_rate
        self.noise_func = noise_function
        self.method = method

    def _explain(self, in_values: torch.Tensor, labels: torch.Tensor, **kwargs) -> np.ndarray:
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
        saliency = np.zeros_like(in_values)
        for i in range(self.smooth_rate):
            noisy_input = self.noise_func(in_values.clone())
            saliency += self.method.explain(noisy_input, labels)
        saliency /= self.smooth_rate
        return saliency


class Var(CompositeSaliencyMethod):
    """
    Local Explanation Methods for Deep Neural Networks lack Sensitivity to Parameter Values (Adebayo et al. 2018)
    """

    def __init__(self, method: SaliencyMethod, smooth_rate=10, noise_function=Smooth.gaussian_noise):
        """
        Initialize a new Var object.
        :param method: The saliency method to apply smoothing to.
        :param smooth_rate: How many noisy inputs to generate.
        :param noise_function: The function to generate noisy input values.
        """
        super().__init__(method)

        self.smooth_rate = smooth_rate
        self.method = method
        self.noise_func = noise_function

    def _explain(self, in_values: torch.Tensor, labels: torch.Tensor, **kwargs) -> np.ndarray:
        """ Smoothens a saliency method of the input w.r.t. the desired label using the variance.

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
        saliency = np.zeros_like(in_values)
        saliency_sq = np.zeros_like(in_values)
        for i in range(self.smooth_rate):
            noisy_input = self.noise_func(in_values.clone())
            temp_saliency = self.method.explain(noisy_input, labels)
            saliency += temp_saliency
            saliency_sq += np.power(temp_saliency, 2)
        saliency /= self.smooth_rate  # Expected saliency
        saliency_sq /= self.smooth_rate  # sum of squared saliency * p
        return saliency_sq - np.power(saliency, 2)  # variance of saliency


class Guided(CompositeSaliencyMethod):
    """
    Implements Guided* saliency methods, combining Guided Backpropagation with other methods
    """
    def __init__(self, method):
        """ Create a new Guided object.

        Parameters
        ----------
        method : SaliencyMethod
            The method to composite.
        """
        super(Guided, self).__init__(method)
        self.guidedBP = GuidedBackProp(self.method.net)

    def _explain(self, in_values: torch.tensor, labels: torch.Tensor, **kwargs) -> np.ndarray:
        """ Multiply a saliency map of the input w.r.t. the desired label, to it's guided Backpropagation.

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
        guide = self.guidedBP.explain(in_values, labels)
        for hook in self.guidedBP.hooks:
            hook.remove()
        self.guidedBP.hooks = []
        return guide * self.method.explain(in_values, labels)
