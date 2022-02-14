from abc import ABC, abstractmethod

import torch
from torchvision.transforms import GaussianBlur


class NoiseMixin:
    """
    Adds Gaussian noise to an existing baseline
    """
    def __init__(self, sigma, *args, **kwargs):
        self.sigma = sigma
        super().__init__(*args, **kwargs)

    def add_noise(self, tensor):
        return tensor + torch.normal(0, self.sigma, tensor.shape, device=tensor.device)


class Baseline(ABC):
    """
    A baseline for use with algorithms such as integrated gradients, occlusion & more
    """
    def __init__(self):
        super().__init__()

    @abstractmethod
    def get(self, in_values, shape=None):
        raise NotImplementedError("This method needs to be implemented in a subclass.")

    def __call__(self, in_values, shape):
        return self.get(in_values, shape)


class FullBaseline(Baseline):
    """
    A baseline where each pixel is the same value (provided by user)
    """
    def __init__(self, fill_value):
        self.fill_value = fill_value
        super(FullBaseline, self).__init__()

    def get(self, in_values, shape=None):
        if shape is None:
            shape = in_values.shape
        return torch.full(shape, self.fill_value, device=in_values.device, dtype=in_values.dtype)


class NoisyFullMask(NoiseMixin, FullBaseline):
    """
    A baseline where each pixel is the same value and then random noise is added.
    """
    def __init__(self, fill_value, sigma=0.2):
        super().__init__(sigma=sigma, fill_value=fill_value)

    def get(self, in_values, shape=None):
        if shape is None:
            shape = in_values.shape
        return self.add_noise(super(NoisyFullMask, self).get(in_values, shape))


class MeanBaseline(Baseline):
    """A baseline where each pixel is the mean value of all pixels (default: over width, height)"""
    def __init__(self, dims=(2, 3)):
        self.dims = dims
        super(MeanBaseline, self).__init__()

    def get(self, in_values, shape=None):
        if shape is None:
            shape = in_values.shape
        mean = in_values.mean(dim=self.dims, keepdim=True)

        # If we reduce the dimension, then we need to tile in that dimension, otherwise not.
        tiling = [shape[i] if i in self.dims else 1 for i in range(len(shape))]
        return mean.tile(tiling)


class NoisyMeanMask(NoiseMixin, MeanBaseline):
    """A baseline where each pixel is the mean value of all pixels and then random noise is added"""
    def __init__(self, sigma, dims=(2, 3, 4)):
        super().__init__(dims=dims, sigma=sigma)

    def get(self, in_values, shape=None):
        if shape is None:
            shape = in_values.shape
        return self.add_noise(super(NoisyMeanMask, self).get(in_values, shape))


class BlurredBaseline(Baseline):
    """A baseline where the image is blurred using a Gaussian kernel"""
    def __init__(self, kernel_size=11, sigma=5):
        super(BlurredBaseline, self).__init__()
        self.blur = GaussianBlur(kernel_size, sigma)

    def get(self, in_values, shape=None):
        return self.blur(in_values)


class UniformBaseline(Baseline):
    """A baseline where each pixel is chosen from a uniform distribution."""
    def __init__(self, lower=0.0, upper=1.0):
        self.distribution = torch.distributions.Uniform(lower, upper)
        super().__init__()

    def get(self, in_values, shape=None):
        if shape is None:
            shape = in_values.shape
        return self.distribution.sample(shape)
