import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from base import SaliencyMethod

#
#  Visualizing and Understanding Convolutional Networks (Zeiler et al. 2014)
#


class Occlusion(SaliencyMethod):

    def __init__(self, net: nn.Module, smoothed=False, smooth_rate=10):
        super().__init__(net, smoothed, smooth_rate)

    def calculate_mask(self, in_values: torch.Tensor, label: torch.Tensor,
                       occlusion_window: torch.Tensor = torch.zeros((3, 8, 8)), resize: bool = False) -> np.ndarray:

        in_shape = in_values.shape[2:]  # Don't count batch & channels
        occlusion_shape = occlusion_window.shape[1:]
        saliency = torch.zeros((1, 3, in_shape[0] // occlusion_shape[0], in_shape[1] // occlusion_shape[1]))

        initial_score = self.net(in_values).squeeze()[label].item()

        for i in range(0, in_shape[0] // occlusion_shape[0]):
            for j in range(0, in_shape[1] // occlusion_shape[1]):
                occluded = in_values.clone()
                occluded[:, :, i * occlusion_shape[0]:(i + 1) * occlusion_shape[0],
                         j * occlusion_shape[1]:(j + 1) * occlusion_shape[1]] = occlusion_window

                score = self.net(occluded).squeeze()[label].item()

                saliency[:, :, j, i] = initial_score - score

        if resize:
            print(saliency.shape, in_values.shape[1:])
            saliency = F.interpolate(saliency, in_shape)
        saliency.squeeze_()
        return saliency
