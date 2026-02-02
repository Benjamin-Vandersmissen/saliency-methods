from abc import abstractmethod

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import sys

from .base import SaliencyMethod
from .utils import extract_layers, safe_divide, resolve_layer

__all__ = ["_CAM", "CAM", "GradCAM", "ScoreCAM", "GradCAMpp", "AblationCAM", "XGradCAM"]


class _CAM(SaliencyMethod):
    """
    A base class for CAM based methods
    """
    def __init__(self, net: nn.Module, resize: bool = True, target_layer=None, **kwargs):
        """
        Initialize a CAM based saliency method object.
        :param net: The neural network model to use.
        :param resize: Whether to resize the resulting saliency map using bi-linear interpolation
        :param target_layer: The layer to hook. (if not given, resolves to the last convolutional layer)
        :param kwargs: Other arguments.
        """
        super().__init__(net, **kwargs)
        self.resize = resize
        self.target_layer = target_layer
        self.activation_hook = []
        self.conv_out = []
                

    def _find_target_layer(self, layers):
        """ Find the given layer for calculating the activation maps."""

        if isinstance(self.target_layer, str):  # Select a layer by name
            try:
                return resolve_layer(self.net, self.target_layer)
            except ValueError:
                raise Exception("No layer was found with name : " + self.target_layer)

        else:  # Select the last convolutional layer
            return [layer for name, layer in layers if isinstance(layer, nn.Conv2d)][-1]
            

    def _hook_target_layer(self, layer):
        """Hook the last convolutional layer to find its output activations."""

        def _conv_hook(_, inp, outp):
            self.conv_out.append(outp)
        
        self.activation_hook.append(layer.register_forward_hook(_conv_hook))

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

    def _explain(self, in_values: torch.Tensor, labels: torch.Tensor, **kwargs) -> np.ndarray:
        batch_size = in_values.shape[0]
        channels = in_values.shape[1]

        weights = self._get_weights(labels)

        conv_out = self.conv_out.pop()
        saliency = torch.empty((channels, batch_size, *conv_out.shape[2:]))

        saliency[:] = F.relu((weights * conv_out).sum(dim=1))
        saliency = saliency.transpose(0, 1)

        if self.resize:
            saliency = F.interpolate(saliency, in_values.shape[2:], mode='bilinear')

        saliency = saliency.detach().cpu().numpy()
        return saliency

    def _init_hooks(self, in_values):
        if type(self.target_layer) != torch.nn.Module:
            layers = extract_layers(self.net, in_values.shape)
            self.target_layer = self._find_target_layer(layers)
        
        self._hook_target_layer(self.target_layer)

    def explain(self, in_values: torch.Tensor, labels: torch.Tensor, **kwargs) -> np.ndarray:
        """ Calculate Class Activation Mapping for the given input.

        Parameters
        ----------

        in_values : 4D-tensor of shape (batch, channel, width, height)
            The image we want to explain. Only the first in the batch is considered.

        labels : 1D-tensor containing *batch* elements
            The labels we want to explain for.

        Returns
        -------

        4D-numpy.ndarray
            A batch of saliency maps for the given images and labels.

        """
        self._init_hooks(in_values)
        out = self.net(in_values)  # so we can find the hooked value

        return super().explain(in_values, labels, out=out, **kwargs)

    def explain_prediction(self, in_values: torch.Tensor, **kwargs) -> np.ndarray:
        """ Calculate CAM saliency maps for a given input, based on the top-predicted labels.

            Parameters
            ----------
            in_values : 4D-tensor of shape (batch, channel, width, height)
                A batch of images we want to generate saliency maps for.
            Returns
            -------

            4D-numpy.ndarray
                A batch of saliency maps for the images provided and labels predicted.
        """
        self._init_hooks(in_values)
        return super(_CAM, self).explain_prediction(in_values, **kwargs)


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
        last_conv_check = False  # TODO: replace this dummy check with an actual check!
        if last_conv_check:
            print("CAM only works with the last convolutional layer.", file=sys.stderr)
        self.fc_layer: nn.Linear = None

    def _find_fc_layer(self, layers):
        """ Find the linear layer for calculating the activation maps."""
        for name, layer in layers[::-1]:
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
        weights = weights.view((*weights.shape[:2], 1, 1))
        return weights

    def _init_hooks(self, in_values):
        if self.fc_layer is None:
            layers = extract_layers(self.net, in_values.shape)
            self._find_target_layer(layers)
            self._hook_target_layer(self.target_layer)
            self._find_fc_layer(layers)


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
        self.grad_hook = []
        self.grad = []

    def _hook_target_layer(self, layer):
        """Hook the given convolutional layer to find its gradients."""

        def _grad_hook(_, inp, outp):
                self.grad.insert(0, outp[0])
        self.grad_hook.append(layer.register_backward_hook(_grad_hook))
        super(GradCAM, self)._hook_target_layer(layer)

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
        weights = self.grad.pop().mean(dim=(2, 3), keepdim=True)  # Global Average Pool over the feature map
        return weights

    def _explain(self, in_values: torch.Tensor, labels: torch.Tensor, **kwargs) -> np.ndarray:

        if len(self.grad) == 0:  # All gradients are consumed, use backward to generate new gradients
            out = kwargs['out']
            self.net.zero_grad()
            torch.gather(out, 1, labels).sum(dim=1, keepdim=True).backward(torch.ones_like(labels))

        return super(GradCAM, self)._explain(in_values, labels, **kwargs)

    def explain(self, in_values: torch.tensor, labels: torch.Tensor, **kwargs) -> np.ndarray:
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
        in_values.requires_grad_(True)
        return super(GradCAM, self).explain(in_values, labels, **kwargs)

    def explain_prediction(self, in_values: torch.tensor, **kwargs) -> np.ndarray:
        """ Calculate GradCAM saliency maps for a given input, based on the top-predicted labels.

            Parameters
            ----------
            in_values : 4D-tensor of shape (batch, channel, width, height)
                A batch of images we want to generate saliency maps for.
            Returns
            -------

            4D-numpy.ndarray
                A batch of saliency maps for the images provided and labels predicted.
        """
        in_values.requires_grad_(True)
        return super(GradCAM, self).explain_prediction(in_values, **kwargs)


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
        conv_out = self.conv_out[-1]  # Do not pop here, as we need this conv_out also in the actual _explain
        batch_size = self.in_values.shape[0]
        in_channels = self.in_values.shape[1]
        conv_channels = conv_out.shape[1]
        scores = torch.empty(batch_size, conv_channels)

        # Disable hook here, as we need to use the network to calculate the score
        for hook in self.activation_hook:
            hook.remove()
        self.activation_hook = []

        minibatch_size = 64  # TODO : make this a parameter, where -1 is mapped to conv_channels.

        # TODO : can we rewrite this to take a minibatch for each batch at the same time?
        #  -> update scores[:, start:end] = (batch, mini_batch, ..)
        with torch.no_grad():
            for i in range(batch_size):
                masks = F.interpolate(conv_out, self.in_values.shape[2:])[i].unsqueeze(0)
                masks = masks.transpose(0, 1)

                # Normalize mask on a per channel base
                masks -= masks.amin(dim=[2, 3], keepdim=True)
                # Use small epsilon for numerical stability
                denominator = masks.amax(dim=(2, 3), keepdim=True)
                masks = safe_divide(masks, denominator)

                # Duplicate mask for each of the input image channels
                masks = masks.tile(1, in_channels, 1, 1)

                for j in range(0, int(np.ceil(conv_channels/minibatch_size))):
                    start = j*minibatch_size
                    end = start+minibatch_size
                    if end > conv_channels:
                        end = conv_channels
                    scores[i][start:end] = (self.net(masks[start:end]*self.in_values[i].unsqueeze(0))[:, labels[i]] -\
                                           self.base_score[labels[i]]).squeeze()

        # Re-enable hook as we are finished with it.
        self._hook_target_layer(self.target_layer)

        scores = F.softmax(scores, dim=1)
        return scores.reshape((batch_size, conv_channels, 1, 1))

    def _generate_baseline(self, in_values):
        self.in_values = in_values
        baseline = torch.zeros((1, *self.in_values.shape[1:]), device=self.device)
        self.base_score = self.net(baseline).squeeze()

    def explain(self, in_values: torch.tensor, labels: torch.Tensor, **kwargs) -> np.ndarray:
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
        self._generate_baseline(in_values)
        return super().explain(in_values, labels, **kwargs)

    def explain_prediction(self, in_values: torch.tensor, **kwargs) -> np.ndarray:
        """ Calculate ScoreCAM saliency maps for a given input, based on the top-predicted labels.

            Parameters
            ----------
            in_values : 4D-tensor of shape (batch, channel, width, height)
                A batch of images we want to generate saliency maps for.
            Returns
            -------

            4D-numpy.ndarray
                A batch of saliency maps for the images provided and labels predicted.
        """
        self._generate_baseline(in_values)
        return super(ScoreCAM, self).explain_prediction(in_values, **kwargs)


class GradCAMpp(GradCAM):
    """
    Grad-CAM++: Improved Visual Explanations for Deep Convolutional Networks (Chattopadhyay et al. 2017)
    """

    def __init__(self, net: nn.Module,  **kwargs):
        """
        Initialize a new GradCAM++ saliency method object.
        :param net: The neural network model to use.
        :param kwargs: Other arguments.
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
        conv_out = self.conv_out[-1]  # Do not pop here, as we need this conv_out also in the actual _explain
        grad = self.grad.pop()
        grad_2 = torch.pow(grad, 2)
        grad_3 = torch.pow(grad, 3)

        divisor = 2*grad_2 + (conv_out * grad_3).sum(dim=[2, 3], keepdim=True)
        weights = (safe_divide(grad_2, divisor) * F.relu(conv_out)).sum(dim=[2, 3], keepdim=True)
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
        channels = self.conv_out[-1].shape[1]

        current_weights = self.target_layer.weight.clone()
        scores = torch.zeros((channels, batch_size))

        # Disable hook here, as we need to use the network to calculate the score
        for hook in self.activation_hook:
            hook.remove()
        self.activation_hook = []

        initial_score = torch.gather(self.net(self.in_values), 1, self.labels)

        with torch.no_grad():
            for i in range(channels):
                self.target_layer.weight[i] = 0
                current_scores = torch.gather(self.net(self.in_values), 1, self.labels)
                scores[i, :] = safe_divide(initial_score - current_scores, initial_score).squeeze()
                self.target_layer.weight[i, :, :, :] = current_weights[i, :, :, :]

        # Re-enable hook as we are finished with it.
        self._hook_target_layer(self.target_layer)
        return scores.reshape(batch_size, channels, 1, 1)

    def _explain(self, in_values: torch.tensor, labels: torch.Tensor, **kwargs) -> np.ndarray:
        self.in_values = in_values
        self.labels = labels
        return super()._explain(in_values, labels, **kwargs)

    def _init_hooks(self, in_values):
        super()._init_hooks(in_values)
        assert isinstance(self.target_layer, torch.nn.Conv2d)  # Only works on Conv2d
        
        


class XGradCAM(GradCAM):
    """
    Axiom-based Grad-CAM: Towards Accurate Visualization and Explanation of CNNs (Fu et al. 2020)
    """

    def __init__(self, net: nn.Module,  **kwargs):
        """
        Initialize a new XGradCAM saliency method object.
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
        conv_out = self.conv_out[-1]  # Do not pop here, as we need this conv_out also in the actual _explain
        denominator = conv_out.sum(dim=(2, 3), keepdim=True)
        weights = (self.grad.pop() * safe_divide(conv_out, denominator)).sum(dim=(2, 3), keepdim=True)
        return weights


class _MultiCAMMixin:
    def __init__(self, net = None, target_layers=None, **kwargs):
        self.target_layers = target_layers
        self.net = None
        self.grad = None
        
        if self.target_layers is None and 'target_layer' in kwargs:
            print("""Parameter 'target_layer' is set, but as this method combines maps at multiple layers, this parameter is discarded.
Please instead set the 'target_layers' parameter.
As a fallback, the current output will be the aggregate of the maps of all Conv2d layers.""", file=sys.stderr)
        if self.target_layers is not None:
            assert type(self.target_layers) is list
    
    def _init_hooks(self, in_values):
        if self.target_layers is not None:
            for l in self.target_layers:
                self._hook_target_layer(resolve_layer(self.net, l))
        else:
            layers = extract_layers(self.net, in_values.shape)
            for name, layer in layers:
                if isinstance(layer, nn.Conv2d):
                    self._hook_target_layer(layer)
                
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
        # pop the last gradient to calculate the weights
        return self.grad.pop()
    

class ZoomCAM(_MultiCAMMixin, GradCAM):
    """
    Zoom-CAM: Generating Fine-grained Pixel Annotations from Image Labels (Shi et al. 2020)
    """
    def __init__(self, net, **kwargs):
        super().__init__(net, **kwargs)
        super(_MultiCAMMixin, self).__init__(net, **kwargs)
        self.resize = True  # This needs to be True as the intermediary CAM maps need to be scaled up for merging.


    def _explain(self, in_values: torch.tensor, labels: torch.Tensor, **kwargs) -> np.ndarray:
        """
        ZoomCAM is in essence a stacking of Grad-CAM maps at different levels in the network (with point-wise weights).
        To accomplish this, we get the gradients & outputs of all convolutional layers and use the CAM mechanism
        of the superclass to calculate a map, aggregate it with the current temporary map and repeat this process
        until no more maps need to be calculated
        """

        saliency = np.zeros(in_values.shape)

        while len(self.conv_out) > 0:
            new_saliency = super(ZoomCAM, self)._explain(in_values, labels, **kwargs)
            new_saliency = self._normalize(new_saliency)

            saliency = np.maximum(saliency, new_saliency)
        return saliency


class LayerCAM(_MultiCAMMixin, GradCAM):
    """
    LayerCAM: Exploring Hierarchical Activation Maps for Localization (Jiang et al. 2021)
    """
    def __init__(self, net, **kwargs):
        super().__init__(net, **kwargs)
        super(_MultiCAMMixin, self).__init__(net, **kwargs)
        self.resize = True  # This needs to be True as the intermediary CAM maps need to be scaled up for merging.

    def _explain(self, in_values: torch.tensor, labels: torch.Tensor, **kwargs) -> np.ndarray:

        saliency = np.zeros(in_values.shape)

        while len(self.conv_out) > 0:
            new_saliency = super(LayerCAM, self)._explain(in_values, labels, **kwargs)
            new_saliency = self._postprocess(new_saliency, only_positive=True, pixel_level=True)

            saliency = saliency + new_saliency
        return saliency