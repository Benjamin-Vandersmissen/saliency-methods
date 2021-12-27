from abc import ABC, abstractmethod

import torch
from torchvision.transforms import GaussianBlur


class NoiseMixin:
    def __init__(self, sigma, *args, **kwargs):
        self.sigma = sigma
        super().__init__(*args, **kwargs)

    def add_noise(self, tensor):
        return tensor + torch.normal(0, self.sigma, tensor.shape, device=tensor.device)


class Mask(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def mask(self, in_values, shape=None):
        raise NotImplementedError("This method needs to be implemented in a subclass.")


class FullMask(Mask):
    def __init__(self, fill_value):
        self.fill_value = fill_value
        super(FullMask, self).__init__()

    def mask(self, in_values, shape=None):
        if shape is None:
            shape = in_values.shape
        return torch.full(shape, self.fill_value, device=in_values.device, dtype=in_values.dtype)


class NoisyFullMask(NoiseMixin, FullMask):
    def __init__(self, fill_value, sigma=0.2):
        super().__init__(sigma=sigma, fill_value=fill_value)

    def mask(self, in_values, shape=None):
        if shape is None:
            shape = in_values.shape
        return self.add_noise(super(NoisyFullMask, self).mask(in_values, shape))


class MeanMask(Mask):
    def __init__(self, dims=(1, 2, 3)):
        self.dims = dims
        super(MeanMask, self).__init__()

    def mask(self, in_values, shape=None):
        if shape is None:
            shape = in_values.shape
        mean = in_values.mean(dim=self.dims, keepdim=True)

        # If we reduce the dimension, then we need to tile in that dimension, otherwise not.
        tiling = [shape[i] if i in self.dims else 1 for i in range(len(shape))]
        return mean.tile(tiling)


class NoisyMeanMask(NoiseMixin, MeanMask):
    def __init__(self, sigma, dims=(2, 3, 4)):
        super().__init__(dims=dims, sigma=sigma)

    def mask(self, in_values, shape=None):
        if shape is None:
            shape = in_values.shape
        return self.add_noise(super(NoisyMeanMask, self).mask(in_values, shape))


class BlurredMask(Mask):
    def __init__(self, kernel_size=3, sigma=(0.1, 2)):
        super(BlurredMask, self).__init__()
        self.blur = GaussianBlur(kernel_size, sigma)

    def mask(self, in_values, shape=None):
        return self.blur(in_values)
