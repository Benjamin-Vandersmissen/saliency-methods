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
    def calculate_map(self, in_values: torch.tensor, labels: torch.Tensor, **kwargs) -> np.ndarray:
        """ Calculate a saliency map for the given input.

        Parameters
        ----------

        in_values : 4D-tensor of shape (batch, channel, width, height)
            The image we want to explain. Only the first in the batch is considered.

        labels : 1D-tensor containing *batch* elements
            The labels we want to explain for.

        Returns
        -------

        4D-numpy.ndarray
            A batch of saliency maps for the images and labels provided.

        """
        raise NotImplementedError("A Subclass of SaliencyMethod needs to implement this function")


class CompositeSaliencyMethod(SaliencyMethod):

    def __init__(self, method: SaliencyMethod):
        """ Create a new CompositeSaliencyMethod object.

        Parameters
        ----------
        method : SaliencyMethod
            The method to composite.
        """

        super().__init__(method.net, method.device)
        self.method = method

    @abstractmethod
    def calculate_map(self, in_values: torch.tensor, labels: torch.Tensor, **kwargs) -> np.ndarray:
        """ Calculate a composite saliency map for the given input by combining multiple methods.

        Parameters
        ----------

        in_values : 4D-tensor of shape (batch, channel, width, height)
            The image we want to explain. Only the first in the batch is considered.

        labels : 1D-tensor containing *batch* elements
            The labels we want to explain for.

        Returns
        -------

        4D-numpy.ndarray
            A batch of saliency maps for the images and labels provided.

        """
        return self.method.calculate_map(in_values, labels, **kwargs)
