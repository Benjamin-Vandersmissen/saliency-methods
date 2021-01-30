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

    def calculate_mask(self, in_values: torch.Tensor, label: torch.Tensor, baseline=None, nr_steps=50) -> np.ndarray:
        if baseline is None:
            baseline = torch.zeros_like(in_values)

        gradients = []

        for i in range(1, nr_steps+1):
            current_input = baseline + (i/nr_steps) * (in_values - baseline)
            gradients.append(super(IntegratedGradient, self).calculate_mask(current_input, label))

        saliency = ((in_values-baseline)*np.average(gradients)).squeeze().detach().numpy()
        return saliency
