import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .base import SaliencyMethod

__all__ = ['Occlusion']


class Occlusion(SaliencyMethod):
    """ Calculate the occlusion map of the input w.r.t. the desired label.

    Parameters
    ----------

    net : torch.nn.module
        The network used for generating a saliency map.

    smoothed : bool
        Whether to apply smoothing via SMOOTHGRAD (True) or not (False).

    smooth_rate : int
        How many iterations of SMOOTHGRAD to use.

    mgf : function
        A Mask Generating Function, which accepts the image and the size and outputs a mask of the correct size.
        If occlusion_window is set, this parameter does nothing

    occlusion_size : 3D-tuple of shape (channel, width, height)
        The size of the occlusion window.
        If occlusion_window is set, this parameter does nothing

    occlusion_window : torch.Tensor
        If set, don't generate a mask for each different saliency map, but reuse this value.

    resize : bool
        Resize the map via nearest neighbours if True, otherwise don't resize.

    References
    ----------

    Visualizing and Understanding Convolutional Networks (Zeiler et al. 2014)

    """

    @staticmethod
    def uniform_mask(image: torch.tensor, shape: tuple, value:float = 0):
        mask = torch.ones(shape)*value
        return mask

    @staticmethod
    def mean_mask(image: torch.tensor, shape:tuple):
        mean = image.mean()
        return torch.ones(shape)*mean

    def __init__(self, net: nn.Module, smoothed=False, smooth_rate=10, mgf=mean_mask.__func__, occlusion_size=(3, 8, 8),
                 occlusion_window: torch.Tensor = None, resize: bool = False):

        self.mgf = mgf
        self.occlusion_size = occlusion_size
        self.occlusion_window = occlusion_window
        self.resize = resize
        super().__init__(net, smoothed, smooth_rate)

    def _calculate(self, in_values: torch.Tensor, label: torch.Tensor, **kwargs) -> np.ndarray:

        """ Calculates the Occlusion map of the input w.r.t. the desired label.

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
        occlusion_window = self.occlusion_window
        if occlusion_window is None:
            occlusion_window = self.mgf(in_values, self.occlusion_size)

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

        if self.resize:
            saliency = F.interpolate(saliency, in_shape)
        saliency.squeeze_().detach().numpy()
        return saliency
