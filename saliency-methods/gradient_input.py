from torch import nn
import torch
import numpy as np

from gradient import Gradient

#
#  Learning Important Features Through Propagating Activation Differences (Shrikumar et al. 2017)
#  https://arxiv.org/pdf/1312.6034.pdf
#


class GradientXInput(Gradient):

    def __init__(self, net: nn.Module, smoothed=False, smooth_rate=10):
        super(Gradient, self).__init__(net, smoothed, smooth_rate)

    def calculate_mask(self, in_values: torch.Tensor, label: torch.Tensor) -> np.ndarray:

        gradient = super().calculate_mask(in_values, label)
        saliency = gradient * in_values.squeeze().detach().numpy()

        return saliency
