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
        """Normalize a batch of saliency maps to range [0,1].

        Parameters
        ----------

        saliency: 4D-np.ndarray of shape (batch, channel, width, height)
            The calculated batch of saliency maps.

        Returns
        -------

        4D-np.ndarray of shape (batch, channel, width, height)
            The batch of saliency maps normalized over (channel, width, height).
        """
        if len(saliency.shape) == 4:
            axis = (1, 2, 3)
        elif len(saliency.shape) == 3:
            axis = (1, 2)
        min = saliency.min(axis=axis, keepdims=True)
        max = saliency.max(axis=axis, keepdims=True)
        return (saliency - min) / (max-min)

    @staticmethod
    def _postprocess(saliency: np.ndarray, only_positive=False, normalize=False, pixel_level=False, **kwargs) -> np.ndarray:
        """Postprocess a batch of saliency maps.

        Parameters
        ----------

        saliency: 4D-np.ndarray of shape (batch, channel, width, height)
            The calculated batch of saliency maps.

        only_positive: bool
            Whether to allow only positive relevance or also negative relevance.

        normalize: bool
            Whether to normalize the resulting saliency map or not.

        pixel_level: bool
            Whether the saliency map contains relevance per pixel, or per channel x pixel.
        Returns
        -------

        4D-np.ndarray of shape (batch, channel, width, height)
            The batch of saliency maps postprocessed.
        """
        if only_positive:
            saliency = np.maximum(0, saliency)

        if pixel_level:
            saliency = saliency.mean(axis=1, keepdims=True)

        if normalize:
            saliency = SaliencyMethod._normalize(saliency)

        return saliency

    @abstractmethod
    def _explain(self, in_values: torch.Tensor, labels: torch.Tensor, **kwargs) -> np.ndarray:
        raise NotImplementedError("A Subclass of SaliencyMethod needs to implement this method")

    def explain(self, in_values: torch.tensor, labels: torch.Tensor, **kwargs) -> np.ndarray:
        """ Calculate a saliency map for the given input.

        Parameters
        ----------

        in_values : 4D-tensor of shape (batch, channel, width, height)
            A batch of images we want to generate saliency maps for.

        labels : 1D-tensor containing *batch* elements
            The labels we want to explain for.

        Returns
        -------

        4D-numpy.ndarray
            A batch of saliency maps for the images and labels provided.

        """
        labels = labels.reshape((labels.shape[0], 1)).to(self.device)
        in_values = in_values.to(self.device)
        return self._postprocess(self._explain(in_values, labels, **kwargs), **kwargs)

    def explain_prediction(self, in_values: torch.tensor, **kwargs) -> np.ndarray:
        """ Calculate saliency maps for a given input, based on the top-predicted labels.

        Parameters
        ----------
        in_values : 4D-tensor of shape (batch, channel, width, height)
            A batch of images we want to generate saliency maps for.
        Returns
        -------

        2D-numpy.ndarray
            The predicted labels for the input images.
        4D-numpy.ndarray
            A batch of saliency maps for the images provided and labels predicted.
        """
        in_values = in_values.to(self.device)
        out = self.net(in_values)
        labels = torch.argmax(out, dim=1, keepdim=True)
        return labels.cpu().numpy(), self._postprocess(self._explain(in_values, labels, out=out), **kwargs)  # In case we need the prediction scores.

    
    def __call__(self, *args, **kwargs):
        return self.explain(*args,**kwargs)

    def __enter__(self):
        return self
    
    def __exit__(self):
        pass

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

    def _explain(self, in_values: torch.Tensor, labels: torch.Tensor, **kwargs) -> np.ndarray:
        return self.method._explain(in_values, labels, **kwargs)
