import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .base import SaliencyMethod
from .utils import extract_layers

__all__ = ["_CAM", "CAM", "GradCAM"]


class _CAM(SaliencyMethod):
    """ A base class for CAM based methods

    Parameters
    ----------
    resize : bool
        Whether to resize the map (True) or not (False).

    normalise : bool
        Whether to normalise the map (True) or not (False).
    """
    def __init__(self, net: nn.Module, resize: bool = True, normalise: bool = True, **kwargs):
        super().__init__(net, **kwargs)
        self.resize = resize
        self.normalised = normalise
        self.conv_layer: nn.Conv2d = None
        self.activation_hook = None
        self.conv_out = None
        self.layers = None

    def _find_conv_layer(self):
        """ Find the last convolutional layer for calculating the activation maps."""
        for layer in self.layers[::-1]:
            if isinstance(layer, nn.Conv2d):
                self.conv_layer = layer
                break

    def _hook_conv_layer(self):
        """Hook the last convolutional layer to find its output activations."""

        def _conv_hook(_, __, outp):
            self.conv_out = outp

        if self.activation_hook is None:
            self.activation_hook = self.conv_layer.register_forward_hook(_conv_hook)

    def _get_weights(self, label: torch.Tensor) -> torch.Tensor:
        """Get the weights used for the CAM calculations

        Parameters
        ----------

        label : 1D-tensor that contains the labels
            The labels for which we want to calculate the weights.

        Returns
        -------

        weights : 4D-tensor of shape (batch, channel, width, height)
            The weights used in the linear combination of activation maps.

        """
        raise NotImplementedError

    def calculate_map(self, in_values: torch.tensor, labels: torch.Tensor, **kwargs) -> np.ndarray:
        """ Calculate Class Activation Mapping for the given input.

        Parameters
        ----------

        in_values : 4D-tensor of shape (batch, channel, width, height)
            The image we want to explain. Only the first in the batch is considered.

        labels : 1D-tensor containing *batch* elements
            The labels we want to explain for.

        Returns
        -------

        3D-numpy.ndarray
            A saliency map for the first image in the batch.

        """
        in_values = in_values.to(self.device)
        batch_size = in_values.shape[0]
        channels = in_values.shape[1]

        if self.layers is None:
            self.layers = extract_layers(self.net, in_values.shape)
            self._find_conv_layer()
            self._hook_conv_layer()

        _ = self.net(in_values)  # so we can find the hooked value

        weights = self._get_weights(labels)

        saliency = torch.empty((channels, batch_size, *self.conv_out.shape[2:]))

        saliency[:] = F.relu((weights * self.conv_out).sum(dim=1))
        saliency = saliency.transpose(0, 1)

        if self.resize:
            saliency = F.interpolate(saliency, in_values.shape[2:], mode='bilinear')

        saliency = saliency.detach().numpy()

        if self.normalised:
            saliency = self._normalize(saliency)

        return saliency


class CAM(_CAM):
    """ Calculate the Class Activation Map of the input w.r.t. the desired label.

    Parameters
    ----------

    net : torch.nn.module
        The network used for generating a saliency map.

    resize : bool
        Resize the map via bi-linear interpolation if True, otherwise don't resize.

    References
    ----------

    Learning Deep Features for Discriminative Localization (Zhou et al. 2016)

    """

    def __init__(self, net: nn.Module, **kwargs):
        super().__init__(net, **kwargs)
        self.fc_layer: nn.Linear = None

    def _find_fc_layer(self):
        """ Find the linear layer for calculating the activation maps."""
        for layer in self.layers[::-1]:
            if isinstance(layer, nn.Linear):
                self.fc_layer = layer
                break

    def _get_weights(self, labels: torch.Tensor) -> torch.Tensor:
        """ Get the weights used for the CAM calculations.

        Parameters
        ----------

        labels : 1D-tensor that contains the labels
            The labels for which we want to calculate the weights.

        Returns
        -------

        weights : 4D-tensor of shape (channel, width, height)
            The weights used in the linear combination of activation maps.

        """
        weights = self.fc_layer.weight[labels]
        weights = weights.view((*weights.shape[0:2], 1, 1))
        return weights

    def calculate_map(self, in_values: torch.tensor, labels: torch.Tensor, **kwargs) -> np.ndarray:
        """ Calculate Class Activation Mapping for the given input.

        Parameters
        ----------

        in_values : 4D-tensor of shape (batch, channel, width, height)
            The image we want to explain. Only the first in the batch is considered.

        labels : 1D-tensor
            The labels we want to explain for.

        Returns
        -------

        4D-numpy.ndarray
            A saliency map for the first image in the batch.

        """
        if self.layers is None:
            self.layers = extract_layers(self.net, in_values.shape)
            self._find_conv_layer()
            self._hook_conv_layer()
            self._find_fc_layer()

        return super().calculate_map(in_values, labels, **kwargs)


class GradCAM(_CAM):
    """ Calculate GradCAM for the input w.r.t. the desired label.

    Parameters
    ----------

    net : torch.nn.module
        The network used for generating a saliency map.

    resize : bool
        Resize the map via bi-linear interpolation if True, otherwise don't resize.

    References
    ----------

    Grad-CAM: Visual explanations from deep networks via gradient-based localization (Selvaraju et al. 2017)

    """

    def __init__(self, net: nn.Module,  **kwargs):
        super().__init__(net, **kwargs)
        self.grad_hook = None
        self.grad = None

    def _hook_grad(self):
        """Hook the last convolutional layer to find its gradients."""

        def _grad_hook(_, __, outp):
            self.grad = outp[0]
        if self.grad_hook is None:
            self.grad_hook = self.conv_layer.register_backward_hook(_grad_hook)

    def _get_weights(self, label: torch.Tensor) -> torch.Tensor:
        """ Get the weights used for the CAM calculations

        Parameters
        ----------

        label : 1D-tensor that contains the labels
            The labels for which we want to calculate the weights.

        Returns
        -------

        weights : 4D-tensor of shape (batch, channel, width, height)
            The weights used in the linear combination of activation maps.

        """
        weights = self.grad.mean(dim=(2, 3))  # Global Average Pool over the feature map
        weights = weights.view((*weights.shape[0:2], 1, 1))
        return weights

    def _backprop(self, in_values: torch.Tensor, labels: torch.Tensor):
        """Backpropagate the score of the input w.r.t. the expected label.

        Parameters
        ----------

        in_values : 4D-tensor of shape (batch, channel, width, height)
            Input values to backpropagate.

        labels : 1D-tensor
            Label to backpropagate for.
        """

        labels = labels.view((1, labels.shape[0]))
        in_values.requires_grad_(True)
        scores = self.net(in_values)
        self.net.zero_grad()
        torch.gather(scores, 1, labels).sum(dim=0, keepdim=True).backward(torch.ones_like(labels))

    def calculate_map(self, in_values: torch.tensor, label: torch.Tensor, resize: bool = True, **kwargs) -> np.ndarray:
        """ Calculate Class Activation Mapping for the given input.

        Parameters
        ----------

        in_values : 4D-tensor of shape (batch, channel, width, height)
            The images we want to explain.

        label : 1D-tensor
            The labels we want to explain for.

        resize : boolean, default=True
            Resize the saliency map using bilinear interpolation.

        Returns
        -------

        4D-numpy.ndarray
            A batch of saliency maps.

        """
        in_values.to(self.device)

        if self.layers is None:
            self.layers = extract_layers(self.net, in_values.shape)
            self._find_conv_layer()
            self._hook_conv_layer()
            self._hook_grad()

        self._backprop(in_values, label)

        return super().calculate_map(in_values, label, **kwargs)
