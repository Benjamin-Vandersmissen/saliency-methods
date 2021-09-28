import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import random

from .base import SaliencyMethod

__all__ = ['Rise']


class Rise(SaliencyMethod):
    """ Calculate the saliency map of the input w.r.t. the desired label.

    Parameters
    ----------

    net : torch.nn.module
        The network used for generating a saliency map.

    nr_masks : int
        How many masks are used for the RISE algorithm (These are reused for each explanation.


    References
    ----------

    Rise : Randomized Input Sampling for Explanation (Petsiuk et al. 2018)

    """

    def __init__(self, net: nn.Module, nr_masks=2500, sample_size=(7, 7), p=0.1, **kwargs):
        self.nr_masks = nr_masks
        self.sample_size = sample_size
        self.p = p
        self.masks = None
        super().__init__(net, **kwargs)

    def _generate_mask(self, image_size):
        """ Generates input masks according to the Rise Paper

        Parameters
        ----------

        image_size : tuple of shape (width, height)
            The dimensions the mask needs to have.
        """
        # generate mask of size (sample_width, sample_height) where each value is 1 witch probability p,
        # 0 with probability 1-p
        sample_mask = torch.bernoulli(torch.full(self.sample_size, self.p))

        new_width = int((sample_mask.shape[0]+1)*np.ceil(image_size[0]/sample_mask.shape[0]))
        new_height = int((sample_mask.shape[1]+1)*np.ceil(image_size[1]/sample_mask.shape[1]))

        # needed for F.interpolate, as it needs a (batch, channel, width, height) Tensor
        sample_mask = sample_mask.view((1, 1, *sample_mask.shape))
        mask = F.interpolate(sample_mask, (new_width, new_height), mode="bilinear").squeeze()

        # Randomly shift the boundaries such that the mask size is equal to the image size.
        delta_w = random.randint(0, mask.shape[0] - image_size[0])
        delta_h = random.randint(0, mask.shape[1] - image_size[1])

        mask = mask[delta_w:image_size[0] + delta_w, delta_h:image_size[1] + delta_h].detach().numpy()

        return mask

    def calculate_map(self, in_values: torch.Tensor, label: torch.Tensor, **kwargs) -> np.ndarray:

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
        in_values = in_values.to(self.device)

        image_size = in_values.shape[2:]
        if self.masks is None:
            self.masks = []
            for i in range(self.nr_masks):
                self.masks.append(self._generate_mask(image_size))
            self.masks = np.asarray(self.masks)

        scores = np.empty(self.nr_masks)
        for i in range(self.nr_masks):
            mask = self.masks[i]
            masked_in = (in_values * mask).to(self.device)
            scores[i] = F.softmax(self.net(masked_in), dim=1).squeeze()[label].item()

        saliency = np.empty((3, *image_size))
        saliency[:] = (1 / (self.p * self.nr_masks) * scores.reshape((self.nr_masks, 1, 1)) * self.masks).sum(axis=0)

        saliency = self._normalize(saliency)
        return saliency
