from torch import nn
import torch
import numpy as np

from base import SaliencyMethod

#
#  Deep inside convolutional networks: Visualising image classification models and saliency maps. (Simonyan et al. 2013)
#  https://arxiv.org/pdf/1312.6034.pdf
#


class Gradient(SaliencyMethod):

    def __init__(self, net: nn.Module, smoothed=False, smooth_rate=10):
        super(Gradient, self).__init__(net, smoothed, smooth_rate)

    def calculate_mask(self, in_values: torch.Tensor, label: torch.Tensor) -> np.ndarray:

        in_values = in_values.data.requires_grad_(True)
        self.net.zero_grad()

        out_values = self.net(in_values)[:, label]  # select relevant score
        out_values.backward()
        saliency = in_values.grad.squeeze().detach().numpy()

        return saliency
