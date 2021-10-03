import sys

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

    mgf : function
        A Mask Generating Function, which accepts the image and the size and outputs a mask of the correct size.
        If occlusion_window is set, this parameter does nothing

    occlusion_size : 3D-tuple of shape (channel, width, height)
        The size of the occlusion window. This should fit exactly in the input images.
        If occlusion_window is set, this parameter does nothing

    occlusion_window : torch.Tensor
        If set, don't generate a mask for each different saliency map, but reuse this value.
        The dimensions of this window should fit exactly in the input images.

    resize : bool
        Resize the map to the image size via nearest neighbours interpolation if True, otherwise don't resize.

    Brief
    -----

    Create a saliency map by systematically occluding a part of an image and calculating the difference between
    the prediction score of the un-occluded image and occluded image. This difference is then the importance score
    of the occluded patch. If resize = False, then the resulting feature map will have size
    (#channels, image_width//window_width, image_height//window_height), else the feature map will be rescaled to the
    size of the input image.

    References
    ----------

    Visualizing and Understanding Convolutional Networks (Zeiler et al. 2014)

    """

    @staticmethod
    def uniform_mask(image: torch.tensor, shape: tuple, value: float = 0):
        return torch.full(shape, value)

    @staticmethod
    def mean_mask(image: torch.tensor, shape: tuple):
        return image.mean(dim=[1, 2, 3], keepdim=True).tile((1, *shape))

    def __init__(self, net: nn.Module, mgf=mean_mask.__func__, occlusion_size=(3, 8, 8), occlusion_window: torch.Tensor = None, resize: bool = False, **kwargs):

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
            The image(s) we want to explain.

        labels : 1D-tensor of *batch* elements
            The label we want to explain for.

        Returns
        -------

        4D-numpy.ndarray
            A batch of saliency maps.

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
        saliency = saliency.detach().numpy()
        return saliency
