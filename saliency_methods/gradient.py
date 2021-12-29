from torch import nn
import torch
import torch.nn.functional as F
import numpy as np

from .base import SaliencyMethod
from .utils import EPSILON
from .mask import Mask, MeanMask
__all__ = ['Gradient', 'GradientXInput', 'IntegratedGradients', 'FullGradient', 'GuidedBackProp']


class Gradient(SaliencyMethod):
    """
    Deep inside convolutional networks: Visualising image classification models and saliency maps. (Simonyan et al. 2013)
    """

    def __init__(self, net: nn.Module, **kwargs):
        """
        Initialize a Gradient Saliency Method object.
        :param net: The neural network to use.
        :param kwargs: Other parameters.
        """
        super(Gradient, self).__init__(net, **kwargs)

    def _explain(self, in_values: torch.Tensor, labels: torch.Tensor, **kwargs) -> np.ndarray:
        out_values = torch.gather(self.net(in_values), 1, labels)  # select relevant score
        out_values.backward(gradient=torch.ones_like(out_values))
        saliency = in_values.grad.detach().cpu().numpy()
        return saliency

    def explain(self, in_values: torch.tensor, labels: torch.Tensor, **kwargs) -> np.ndarray:
        """ Calculates the Gradient of the input w.r.t. the desired label.

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
        in_values = in_values.data.requires_grad_(True)
        self.net.zero_grad()

        return super(Gradient, self).explain(in_values, labels, **kwargs)

    def explain_prediction(self, in_values: torch.tensor, **kwargs) -> np.ndarray:
        in_values = in_values.data.requires_grad_(True)
        self.net.zero_grad()
        return super(Gradient, self).explain_prediction(in_values, **kwargs)


class GradientXInput(Gradient):
    """
    Learning Important Features Through Propagating Activation Differences (Shrikumar et al. 2017)
    """

    def __init__(self, net: nn.Module, **kwargs):
        """
        Initialize a GradientxInput Saliency Method object.
        :param net: The neural network to use.
        :param kwargs: Other parameters.
        """
        super(Gradient, self).__init__(net, **kwargs)

    def _explain(self, in_values: torch.Tensor, labels: torch.Tensor, **kwargs) -> np.ndarray:
        return super(GradientXInput, self)._explain(in_values, labels) * in_values.detach().cpu().numpy()


class IntegratedGradients(Gradient):
    """
    Axiomatic Attribution for Deep Networks (Sundararajan et al. 2017)
    """

    def __init__(self, net: nn.Module,  baseline: Mask = MeanMask(), nr_steps=50, **kwargs):
        """
        Initialize a Integrated Gradients Saliency Method object.
        :param net: The neural network to use.
        :param baseline: Which type of baseline to use.
        :param nr_steps: In how many steps do we integrate
        :param kwargs: Other parameters.
        """
        super().__init__(net, **kwargs)
        self.baseline = baseline
        self.nr_steps = nr_steps

    def get_baseline(self, in_values):
        """
        Generate a baseline image for the input values.
        :param in_values: A batch of images.
        :return: A batch of baselines, corresponding to the batch of images.
        """
        baseline = self.baseline.mask(in_values, in_values.shape)

        return baseline

    def _explain(self, in_values: torch.Tensor, labels: torch.Tensor, **kwargs) -> np.ndarray:
        baseline = self.get_baseline(in_values)
        grad = np.zeros(in_values.shape)

        for i in range(1, self.nr_steps + 1):
            current_input = (baseline + (i / self.nr_steps) * (in_values - baseline)).data.requires_grad_(True)
            grad += super(IntegratedGradients, self)._explain(current_input, labels, **kwargs)

        grad /= self.nr_steps
        in_values = in_values.detach().cpu().numpy()
        baseline = baseline.detach().cpu().numpy()
        saliency = (in_values - baseline) * grad
        return saliency


class FullGradient(Gradient):
    """
    Full-Gradient Representation for Neural Network Visualization (Srinivas et al. 2019)
    """
    def __init__(self, net: nn.Module, **kwargs):
        """
        Initialize a FullGradient Saliency Method object.
        :param net: The neural network to use.
        :param kwargs: extra arguments.
        """
        super().__init__(net, **kwargs)

        self.conv_biases = []
        self.conv_gradients = []

        self.hooks = []
        for component in self.net.modules():
            if isinstance(component, nn.Conv2d):
                self.hooks.append(component.register_backward_hook(self._grad_bias_hook))

    def _grad_bias_hook(self, module, _, out_grad):
        """
        A hook that extracts biases and gradients from conv2d layers and stores them in seperate lists.
        :param module: A nn module
        :param _: /
        :param out_grad: The backwards gradients.
        :return: /
        """
        if module.bias is not None:
            self.conv_gradients.append(out_grad[0])
            self.conv_biases.append(module.bias.data)

    @staticmethod
    def _postprocess_gradient(gradient, shape):
        """
        Post process a gradient by first taking the absolute value,
        then normalising to the [0,1] range and finally upscaling via bilinear interpolation.
        :param gradient: The gradient to post process
        :param shape: The shape to resize to
        :return: A postprocessed gradient.
        """
        gradient = torch.abs(gradient)

        # Normalize to [0,1]
        gradient -= gradient.amin(dim=(2, 3), keepdim=True)
        # Use small epsilon for numerical stability
        denominator = gradient.amax(dim=(2, 3), keepdim=True)
        denominator[denominator == 0] = torch.FloatTensor([EPSILON])
        gradient /= denominator

        # Resize to the correct size.
        gradient = F.interpolate(gradient, shape[2:], mode='bilinear', align_corners=True)
        return gradient

    def _explain(self, in_values: torch.tensor, labels: torch.Tensor, **kwargs) -> np.ndarray:
        shape = in_values.shape

        grad = torch.tensor(super()._explain(in_values, labels), device=self.device)
        saliency = self._postprocess_gradient(grad * in_values, shape)

        while len(self.conv_gradients) > 0:
            grad = self.conv_gradients.pop()
            bias = self.conv_biases.pop()
            processed = self._postprocess_gradient(grad * bias.view(1, -1, 1, 1), shape)
            processed = processed.sum(dim=1, keepdims=True)
            saliency += processed

        return saliency.detach().cpu().numpy()


class GuidedBackProp(Gradient):
    """
    Striving for simplicity, the all convolutional net (Springenberg et al. 2014)
    """
    def __init__(self, net, **kwargs):
        """
        Initialize a GuidedBackProp Saliency Method object.
        :param net: The neural network to use.
        :param kwargs: extra arguments.
        """
        super(GuidedBackProp, self).__init__(net)
        self.hooks = []
        self.hook_grad()

    def hook_grad(self):
        """
        Add forward and backward hooks to the ReLU modules, to clip their gradients to positive values.
        :return: /
        """
        def back_hook(module, in_grad, out_grad):
            outp = module.outp.pop()
            if len(module.outp) == 0:
                del module.outp
            out = F.relu((outp != 0) * out_grad[0])
            return out,

        def forward_hook(module, inp, outp):
            if hasattr(module, 'outp'):
                module.outp.append(outp)
            else:
                module.outp = [outp]

        for module in self.net.modules():
            if isinstance(module, nn.ReLU):
                self.hooks.append(module.register_backward_hook(back_hook))
                self.hooks.append(module.register_forward_hook(forward_hook))

    def _explain(self, in_values: torch.tensor, labels: torch.Tensor, **kwargs) -> np.ndarray:
        return super(GuidedBackProp, self)._explain(in_values, labels, **kwargs)
