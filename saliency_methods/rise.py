import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import random

from .base import SaliencyMethod

__all__ = ['Rise']


class Rise(SaliencyMethod):
    """
    Rise : Randomized Input Sampling for Explanation (Petsiuk et al. 2018)
    """

    def __init__(self, net: nn.Module, nr_masks=2500, mask_size=(7, 7), p=0.1, **kwargs):
        """
        Initialize a new RISE saliency method object.
        :param net: The neural network to use.
        :param nr_masks: The number of masks to use during saliency map generation.
        :param mask_size: The size of the sampled masks.
        :param p: The percentage of 1's in the bernoulli distributed masks.
        :param kwargs: Other arguments.
        """
        self.nr_masks = nr_masks
        self.mask_size = mask_size
        self.p = p
        super().__init__(net, **kwargs)

    def _generate_mask(self, image_size):
        """ Generates input masks according to the Rise Paper

        Parameters
        ----------

        image_size : tuple of shape (width, height)
            The dimensions the mask needs to have.
        """
        # Generate mask of size (sample_width, sample_height) where each value is ~ Bernoulli(p)
        sample_mask = torch.bernoulli(torch.full(self.mask_size, self.p))

        new_width = int((sample_mask.shape[0]+1)*np.ceil(image_size[0]/sample_mask.shape[0]))
        new_height = int((sample_mask.shape[1]+1)*np.ceil(image_size[1]/sample_mask.shape[1]))

        # needed for F.interpolate, as it needs a (batch, channel, width, height) Tensor
        sample_mask = sample_mask.view((1, 1, *sample_mask.shape))
        mask = F.interpolate(sample_mask, (new_width, new_height), mode="bilinear").squeeze()

        # Randomly shift the boundaries such that the mask size is equal to the image size.
        delta_w = random.randint(0, mask.shape[0] - image_size[0])
        delta_h = random.randint(0, mask.shape[1] - image_size[1])

        mask = mask[delta_w:image_size[0] + delta_w, delta_h:image_size[1] + delta_h].detach().cpu().numpy()

        return mask

    def _explain(self, in_values: torch.Tensor, labels: torch.Tensor, **kwargs) -> np.ndarray:
        batch_size = in_values.shape[0]
        channels = in_values.shape[1]

        image_size = in_values.shape[2:]

        scores_masks = torch.zeros((batch_size, 1, *image_size))
        with torch.no_grad():
            for i in range(self.nr_masks):
                mask = torch.FloatTensor(self._generate_mask(image_size)).to(self.device)
                masked_in = (in_values * mask)
                scores = torch.gather(F.softmax(self.net(masked_in), dim=1), 1, labels)
                scores_masks[:] += (scores * mask).detach().cpu()

        scores_masks = scores_masks.detach().cpu().numpy()
        saliency = np.empty((channels, batch_size, *image_size))
        saliency[:, :] = (1 / (self.p * self.nr_masks) * scores_masks)

        saliency = self._normalize(saliency.transpose((1, 0, 2, 3)))
        return saliency
