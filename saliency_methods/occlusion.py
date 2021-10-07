import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .base import SaliencyMethod

__all__ = ['Occlusion']


class Occlusion(SaliencyMethod):
    """
    Visualizing and Understanding Convolutional Networks (Zeiler et al. 2014)
    """

    @staticmethod
    def uniform_mask(image: torch.tensor, shape: tuple, value: float = 0):
        return torch.full(shape, value)

    @staticmethod
    def mean_mask(image: torch.tensor, shape: tuple):
        return image.mean(dim=[1, 2, 3], keepdim=True).tile((1, *shape))

    def __init__(self, net: nn.Module, mgf=mean_mask.__func__, occlusion_size=(3, 8, 8), occlusion_window: torch.Tensor = None, resize: bool = False, **kwargs):
        """
        Initialize a new Occlusion Saliency Method object.
        :param net: The neural network to use.
        :param mgf: A function to generate masks for occlusion (will be used if occlusion_window is None)
        :param occlusion_size: The size of the occlusion mask to generate (will be used if occlusion_window is None)
        :param occlusion_window: The occlusion window to use.
        :param resize: Whether to resize the resulting saliency map via nearest-neigbour interpolation.
        :param kwargs: Other arguments.
        """
        self.mgf = mgf
        self.occlusion_size = occlusion_size
        self.occlusion_window = occlusion_window
        self.resize = resize
        super().__init__(net, **kwargs)

    def calculate_map(self, in_values: torch.tensor, labels: torch.Tensor, **kwargs) -> np.ndarray:
        """ Calculates the Occlusion map of the input w.r.t. the desired label.

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

        batch_size = in_values.shape[0]
        channels = in_values.shape[1]
        labels = labels.reshape((batch_size, 1))

        occlusion_window = self.occlusion_window
        if occlusion_window is None:
            occlusion_window = self.mgf(in_values, self.occlusion_size).to(self.device)

        in_shape = in_values.shape[2:]  # Don't count batch & channels
        occlusion_shape = occlusion_window.shape[2:]

        if in_shape[0] % occlusion_shape[0] != 0 or in_shape[1] % occlusion_shape[1]:
            print("The occlusion window (size: {0[0]}, {0[1]}) doesn't fit exactly in the image (size: {1[0]}, {1[1]})."
                  .format(occlusion_shape, in_shape), file=sys.stderr)
            print("This might lead to cut-off data at the edges!", file=sys.stderr)

        saliency = torch.zeros((batch_size, channels,
                                in_shape[0] // occlusion_shape[0], in_shape[1] // occlusion_shape[1]))

        initial_scores = torch.gather(self.net(in_values), 1, labels).cpu()

        with torch.no_grad():
            for i in range(0, in_shape[0] // occlusion_shape[0]):
                for j in range(0, in_shape[1] // occlusion_shape[1]):
                    occluded = in_values.clone().to(self.device)
                    occluded[:, :, i * occlusion_shape[0]:(i + 1) * occlusion_shape[0],
                             j * occlusion_shape[1]:(j + 1) * occlusion_shape[1]] = occlusion_window

                    scores = torch.gather(self.net(occluded), 1, labels).cpu()
                    del occluded
                    saliency[:, :, i, j] = (initial_scores - scores).view(batch_size, 1).repeat(1, channels)

                    # We distribute the saliency equally over the channels, as the original approach occluded the pixels.
                    # This means that we modify all channels in each iteration. If we were to occlude each channel
                    # individually, we could have scores for each channel.

        if self.resize:
            saliency = F.interpolate(saliency, in_shape)
        saliency = saliency.detach().cpu().numpy()
        return saliency
