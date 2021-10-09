import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .base import SaliencyMethod
from .utils import extract_layers, EPSILON

__all__ = ["_CAM", "CAM", "GradCAM", "ScoreCAM", "GradCAMpp", "AblationCAM"]


class _CAM(SaliencyMethod):
    """
    A base class for CAM based methods
    """
    def __init__(self, net: nn.Module, resize: bool = True, normalise: bool = True, **kwargs):
        """
        Initialize a CAM based saliency method object.
        :param net: The neural network model to use.
        :param resize: Whether to resize the resulting saliency map using bi-linear interpolation
        :param normalise: Whether to normalize the map to the [0,1] range
        :param kwargs: Other arguments.
        """
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

    def _get_weights(self, labels: torch.Tensor) -> torch.Tensor:
        """Get the weights used for the CAM calculations

        Parameters
        ----------

        labels : 1D-tensor that contains the labels
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

        saliency = saliency.detach().cpu().numpy()

        if self.normalised:
            saliency = self._normalize(saliency)

        return saliency


class CAM(_CAM):
    """
    Learning Deep Features for Discriminative Localization. (Zhou et al. 2016)
    """

    def __init__(self, net: nn.Module, **kwargs):
        """
        Initialize a CAM Saliency Method object.
        :param net: The neural network to use.
        :param kwargs: Other arguments.
        """
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
    """
    Grad-CAM: Visual explanations from deep networks via gradient-based localization (Selvaraju et al. 2017)
    """

    def __init__(self, net: nn.Module,  **kwargs):
        """
        Initialize a new Grad-CAM saliency method object.
        :param net: The neural network to use.
        :param kwargs: Other arguments.
        """
        super().__init__(net, **kwargs)
        self.grad_hook = None
        self.grad = None

    def _hook_grad(self):
        """Hook the last convolutional layer to find its gradients."""

        def _grad_hook(_, __, outp):
            self.grad = outp[0]
        if self.grad_hook is None:
            self.grad_hook = self.conv_layer.register_backward_hook(_grad_hook)

    def _get_weights(self, labels: torch.Tensor) -> torch.Tensor:
        """ Get the weights used for the CAM calculations

        Parameters
        ----------

        labels : 1D-tensor that contains the labels
            The labels for which we want to calculate the weights.

        Returns
        -------

        weights : 4D-tensor of shape (batch, channel, width, height)
            The weights used in the linear combination of activation maps.

        """
        weights = self.grad.mean(dim=(2, 3), keepdim=True)  # Global Average Pool over the feature map
        return weights

    def _backprop(self, in_values: torch.Tensor, labels: torch.Tensor):
        """Backpropagate the score of the input w.r.t. the expected label.

        Parameters
        ----------

        in_values : 4D-tensor of shape (batch, channel, width, height)
            Input values to backpropagate.

        labels : 1D-tensor
            Labels to backpropagate for.
        """

        labels = labels.view((labels.shape[0], 1))
        in_values.requires_grad_(True)
        scores = self.net(in_values)
        self.net.zero_grad()
        torch.gather(scores, 1, labels).sum(dim=1, keepdim=True).backward(torch.ones_like(labels))

    def calculate_map(self, in_values: torch.tensor, labels: torch.Tensor, **kwargs) -> np.ndarray:
        """ Calculates the gradient-based Class Activation Mapping of the input w.r.t. the desired label.

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
        in_values.to(self.device)

        if self.layers is None:
            self.layers = extract_layers(self.net, in_values.shape)
            self._find_conv_layer()
            self._hook_conv_layer()
            self._hook_grad()

        self._backprop(in_values, labels)

        return super().calculate_map(in_values, labels, **kwargs)


class ScoreCAM(_CAM):
    """
    Score-CAM: Score-Weighted Visual Explanations for Convolutional Neural Networks (Wang et al. 2019)
    """
    def __init__(self, net, **kwargs):
        """
        Initialize a new ScoreCAM Saliency Method object.
        :param net: The neural network model to use.
        :param kwargs: Other arguments.
        """
        super().__init__(net, **kwargs)
        self.in_values = None
        self.labels = None
        self.base_score = None

    def _get_weights(self, labels: torch.Tensor) -> torch.Tensor:
        """ Get the weights used for the CAM calculations

        Parameters
        ----------

        labels : 1D-tensor that contains the labels
            The labels for which we want to calculate the weights.

        Returns
        -------

        weights : 4D-tensor of shape (batch, channel, width, height)
            The weights used in the linear combination of activation maps.

        """
        batch_size = self.in_values.shape[0]
        conv_channels = self.conv_out.shape[1]
        in_channels = self.in_values.shape[1]
        scores = torch.empty(batch_size, conv_channels)

        # Disable hook here, as we need to use the network to calculate the score
        self.activation_hook.remove()
        self.activation_hook = None

        for i in range(batch_size):
            masks = F.interpolate(self.conv_out, self.in_values.shape[2:])[1].unsqueeze(0)
            masks = masks.transpose(0, 1)

            # Normalize mask on a per channel base
            masks -= masks.amin(dim=[2, 3], keepdim=True)
            # Use small epsilon for numerical stability
            denominator = torch.max(masks.amax(dim=(2, 3), keepdim=True))
            denominator[denominator == 0] = EPSILON
            masks /= denominator

            # Duplicate mask for each of the input image channels
            masks = masks.tile(1, in_channels, 1, 1)
            scores[i] = self.net(masks * self.in_values[i].unsqueeze(0))[:, labels[i]] - self.base_score[labels[i]]

        # Re-enable hook as we are finished with it.
        self._hook_conv_layer()

        scores = F.softmax(scores, dim=0)
        return scores.reshape((batch_size, conv_channels, 1, 1))

    def calculate_map(self, in_values: torch.tensor, labels: torch.Tensor, **kwargs) -> np.ndarray:
        """ Calculates the Score-based Class Activation Mapping of the input w.r.t. the desired label.

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
        in_values.to(self.device)

        self.in_values = in_values
        self.labels = labels

        baseline = torch.zeros((1, *self.in_values.shape[1:]))
        self.base_score = self.net(baseline).squeeze()

        return super().calculate_map(in_values, labels, **kwargs)


class GradCAMpp(GradCAM):
    """
    Grad-CAM++: Improved Visual Explanations for Deep Convolutional Networks (Chattopadhyay et al. 2017)
    """

    def __init__(self, net: nn.Module,  **kwargs):
        """
        Initialize a new GradCAM++ saliency method object.
        :param net:
        :param kwargs:
        """
        super().__init__(net, **kwargs)

    def _get_weights(self, labels: torch.Tensor) -> torch.Tensor:
        """ Get the weights used for the CAM calculations

        Parameters
        ----------

        labels : 1D-tensor that contains the labels
            The labels for which we want to calculate the weights.

        Returns
        -------

        weights : 4D-tensor of shape (batch, channel, width, height)
            The weights used in the linear combination of activation maps.

        """
        grad_2 = torch.pow(self.grad, 2)
        grad_3 = torch.pow(self.grad, 3)

        divisor = 2*grad_2 + (self.conv_out * grad_3).sum(dim=[2, 3], keepdim=True)
        divisor[divisor == 0] = EPSILON  # epsilon to avoid numerical instability
        weights = ((grad_2 / divisor) * F.relu(self.conv_out)).sum(dim=[2, 3], keepdim=True)
        return weights


class AblationCAM(_CAM):
    """
    Ablation-CAM: Visual Explanations for Deep Convolutional Network via Gradient-free Localization (Desai & Ramaswamy 2020)
    """
    def __init__(self, net, **kwargs):
        """
        Initialize a new AblationCAM Saliency Method object.
        :param net: The neural network model to use.
        :param kwargs: Other arguments.
        """
        super().__init__(net, **kwargs)
        self.in_values = None
        self.base_score = None
        self.labels = None

    def _get_weights(self, labels: torch.Tensor) -> torch.Tensor:
        """ Get the weights used for the CAM calculations

        Parameters
        ----------

        labels : 1D-tensor that contains the labels
            The labels for which we want to calculate the weights.

        Returns
        -------

        weights : 4D-tensor of shape (batch, channel, width, height)
            The weights used in the linear combination of activation maps.

        """
        batch_size = self.in_values.shape[0]
        channels = self.conv_out.shape[1]

        current_weights = self.conv_layer.weight.clone()
        scores = torch.zeros((channels, batch_size))

        # Disable hook here, as we need to use the network to calculate the score
        self.activation_hook.remove()
        self.activation_hook = None

        initial_score = torch.gather(self.net(self.in_values), 1, self.labels)

        with torch.no_grad():
            for i in range(channels):
                self.conv_layer.weight[i] = 0
                scores[i, :] = ((initial_score - torch.gather(self.net(self.in_values), 1, self.labels)) / (initial_score + EPSILON)).squeeze()
                self.conv_layer.weight[i, :, :, :] = current_weights[i, : , :, :]

        # Re-enable hook as we are finished with it.
        self._hook_conv_layer()
        return scores.reshape(batch_size, channels, 1, 1)

    def calculate_map(self, in_values: torch.tensor, labels: torch.Tensor, **kwargs) -> np.ndarray:
        """ Calculates the Score-based Class Activation Mapping of the input w.r.t. the desired label.

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
        in_values.to(self.device)
        self.in_values = in_values
        self.labels = labels.reshape((in_values.shape[0],1))

        return super().calculate_map(in_values, labels, **kwargs)
