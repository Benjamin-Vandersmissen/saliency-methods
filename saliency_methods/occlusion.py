import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .base import SaliencyMethod
from .mask import Mask, FullMask

__all__ = ['Occlusion']


class Occlusion(SaliencyMethod):
    """
    Visualizing and Understanding Convolutional Networks (Zeiler et al. 2014)
    """

    def __init__(self, net: nn.Module, mgf: Mask = FullMask(0), occlusion_size=(8, 8), stride=-1, **kwargs):
        """
        Initialize a new Occlusion Saliency Method object.
        :param net: The neural network to use.
        :param mgf: A function to generate masks for occlusion (will be used if occlusion_window is None)
        :param occlusion_size: The size of the occlusion mask to generate
        :param stride: The stride of the sliding window (default -1 -> stride = occlusion_size)
        :param kwargs: Other arguments.
        """
        self.mgf = mgf
        self.occlusion_size = occlusion_size
        self.stride = stride
        if stride == -1:
            self.stride = occlusion_size
        super().__init__(net, **kwargs)

    def _explain(self, in_values: torch.tensor, labels: torch.Tensor, **kwargs) -> np.ndarray:

        batch_size = in_values.shape[0]
        channels = in_values.shape[1]

        occlusion_window = self.mgf.mask(in_values, (batch_size, channels, *self.occlusion_size))

        in_shape = in_values.shape[2:]  # Don't count batch & channels
        occlusion_shape = occlusion_window.shape[2:]

        if (in_shape[0] - occlusion_shape[0]) % self.stride[0] != 0 or (in_shape[1] % occlusion_shape[1]) % self.stride[1] != 0:
            print("The occlusion window (size: {0[0]}, {0[1]}) doesn't fit exactly in the image (size: {1[0]}, {1[1]})."
                  .format(occlusion_shape, in_shape), file=sys.stderr)
            print("This might lead to cut-off data at the edges!", file=sys.stderr)

        saliency = torch.zeros_like(in_values, device=self.device)
        divisor = torch.zeros_like(in_values, device=self.device)

        initial_scores = torch.gather(F.softmax(self.net(in_values), dim=1), 1, labels).cpu()

        with torch.no_grad():
            for i in range(0, in_shape[0] - occlusion_shape[0]+1, self.stride[0]):
                for j in range(0, in_shape[1] - occlusion_shape[1]+1, self.stride[1]):
                    occluded = in_values.clone().to(self.device)
                    occluded[:, :, i:i + occlusion_shape[0], j:j + occlusion_shape[1]] = occlusion_window

                    scores = torch.gather(F.softmax(self.net(occluded), dim=1), 1, labels).cpu()
                    del occluded
                    saliency[:, :, i:i + occlusion_shape[0], j:j + occlusion_shape[1]] = (initial_scores - scores)
                    divisor[:, :, i:i+occlusion_shape[0], j:j + occlusion_shape[1]] += 1
                    # We distribute the saliency equally over the channels, as the original approach occluded the pixels.
                    # This means that we modify all channels in each iteration. If we were to occlude each channel
                    # individually, we could have scores for each channel.

        saliency = saliency.detach().cpu().numpy()
        return saliency
