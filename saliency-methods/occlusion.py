import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .base import SaliencyMethod

__all__ = ['Occlusion']


class Occlusion(SaliencyMethod):
    #
    #  Visualizing and Understanding Convolutional Networks (Zeiler et al. 2014)
    #

    def __init__(self, net: nn.Module, smoothed=False, smooth_rate=10):
        super().__init__(net, smoothed, smooth_rate)

    def _calculate(self, in_values: torch.Tensor, label: torch.Tensor,
                   occlusion_window: torch.Tensor = torch.zeros((3, 8, 8)), resize: bool = False,
                   **kwargs) -> np.ndarray:

        """ Calculates the Occlusion map of the input w.r.t. the desired label.

        Parameters
        ----------

        in_values : 4D-tensor of shape (batch, channel, width, height)
            The image we want to explain. Only the first in the batch is considered.

        label : 1D-tensor
            The label we want to explain for.

        occlusion_window : 3D-tensor of shape (channel, width, height), optional
            This is the window that is used for occluding parts of the input.
                The amount of channels needs to be the same as in the input.
                The width and height need to be integer divisors of the input width and height.

        resize : bool, optional
            Resize the occlusion map to the input resolution or not.

        Returns
        -------

        3D-numpy.ndarray
            A saliency map for the first image in the batch.

        """

        in_shape = in_values.shape[2:]  # Don't count batch & channels
        occlusion_shape = occlusion_window.shape[1:]
        saliency = torch.zeros((1, occlusion_window.shape[0],
                                in_shape[0] // occlusion_shape[0], in_shape[1] // occlusion_shape[1]))

        initial_score = self.net(in_values).squeeze()[label].item()

        for i in range(0, in_shape[0] // occlusion_shape[0]):
            for j in range(0, in_shape[1] // occlusion_shape[1]):
                occluded = in_values.clone()
                occluded[:, :, i * occlusion_shape[0]:(i + 1) * occlusion_shape[0],
                         j * occlusion_shape[1]:(j + 1) * occlusion_shape[1]] = occlusion_window

                score = self.net(occluded).squeeze()[label].item()

                saliency[:, :, j, i] = (initial_score - score)/3  # distribute relevance equally over channels

                # We distribute the saliency equally over the channels, as the original approach occluded the pixels.
                # This means that we modify all channels in each iteration. If we were to occlude each channel
                # individually, we could have scores for each channel.

        if resize:
            saliency = F.interpolate(saliency, in_shape)
        saliency.squeeze_().detach().numpy()
        return saliency

    def calculate_map(self, in_values: torch.Tensor, label: torch.Tensor,
                      occlusion_window: torch.Tensor = torch.zeros((3, 8, 8)), resize: bool = False,
                      **kwargs) -> np.ndarray:
        """ Calculates the Occlusion map of the input w.r.t. the desired label. Smoothens the map if necessary.

        Parameters
        ----------

        in_values : 4D-tensor of shape (batch, channel, width, height)
            The image we want to explain. Only the first in the batch is considered.

        label : 1D-tensor
            The label we want to explain for.

        occlusion_window : 3D-tensor of shape (channel, width, height), optional
            This is the window that is used for occluding parts of the input.
                The amount of channels needs to be the same as in the input.
                The width and height need to be integer divisors of the input width and height.

        resize : bool, optional
            Resize the occlusion map to the input resolution or not.

        Returns
        -------

        3D-numpy.ndarray
            A saliency map for the first image in the batch.

        """
        return super().calculate_map(in_values, label, occlusion_window=occlusion_window, resize=resize, **kwargs)
