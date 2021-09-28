import numpy as np
import torch
from torch import nn
from abc import ABC, abstractmethod
import copy

#
#  A base class for the saliency methods
#
#


class SaliencyMethod(ABC):

    def __init__(self, net: nn.Module, device: str = "auto"):
        """ Create a new SaliencyMethod object.

        Parameters
        ----------
        net : torch.nn.module
            The network to calculate saliency maps for.

        device : str, default="auto"
            On which device should we do the operations, if "auto", we use the device of the network
        """

        self.net = copy.deepcopy(net)

        if device == "auto":
            self.device = next(self.net.parameters()).device
        else:
            self.device = device
            self.net.to(device)

        self.net.eval()

    @staticmethod
    def _normalize(saliency: np.ndarray) -> np.ndarray:
        """Normalize a saliency map to range [0,1].

        Parameters
        ----------

        saliency: 3D-np.ndarray of shape (channel, width, height)
            The calculated saliency map.

        Returns
        -------

        3D-np.ndarray of shape (channel, width, height)
            The saliency map normalized over all values.
        """
        return (saliency - saliency.min()) / (saliency.max() - saliency.min())

    @abstractmethod
    def calculate_map(self, in_values: torch.tensor, label: torch.Tensor, **kwargs) -> np.ndarray:
        """ Calculate a saliency map for the given input

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
        raise NotImplementedError("A Subclass of SaliencyMethod needs to implement this function")
