import copy
import types

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import SaliencyMethod
from .gradient import Gradient
from .rule import *
from .mask import Mask, FullMask
from .utils import safe_divide
__all__ = ['LRP']


class LRP(Gradient):
    """
    On Pixel-Wise Explanations for Non-Linear Classifier Decision by Layer-wise Relevance Propagation (Bach et al. 2015)
    Special thanks to:
    https://github.com/wjNam/Relative_Attributing_Propagation for an LRP implementation working with Resnet to compare against
    https://github.com/dmitrysarov/LRP_decomposition for a working LRP implementation using autograd.
    """
    def __init__(self, net: nn.Module, **kwargs):
        """
        Initialize a new LRP Saliency Method object.
        :param net: The neural network to use.
        :param kwargs: Other arguments.
        """
        super().__init__(net, **kwargs)

        self.assign_rules()

    def assign_rules(self):
        """ Assign LRP rules to each layer in the neural network

        Throws
        ------
        Exception if a layer has no associated rule.
        """
        identity_layers = [nn.ReLU, nn.BatchNorm2d, nn.Dropout]
        zero_layers = [nn.AdaptiveAvgPool2d, nn.AvgPool2d, nn.MaxPool2d]
        alpha_beta_layers = [nn.Conv2d, nn.Linear]

        layer_map = {layer: LRPIdentityRule for layer in identity_layers}
        layer_map.update({layer: LRPZeroRule for layer in zero_layers})
        layer_map.update({layer: LRPAlphaBetaRule for layer in alpha_beta_layers})

        initial_layer = True
        for layer in self.net.modules():
            if len(list(layer.children())) == 0:
                if layer.__class__ not in layer_map:
                    raise Exception("There is no rule associated with this layer! (%s)" % layer.__class__)

                setattr(layer, 'forward_orig', layer.forward)
                if initial_layer:
                    setattr(layer, 'forward', types.MethodType(getattr(LRPZbRule, 'forward'), layer))
                    initial_layer = False
                else:
                    setattr(layer, 'forward', types.MethodType(getattr(layer_map[layer.__class__], 'forward'), layer))


class DeepLift(SaliencyMethod):
    """
    Learning Important Features Through Propagating Activation Differences (Shrikumar et al. 2017)
    """
    def __init__(self, net: nn.Module, baseline_fn: Mask = FullMask(0), **kwargs):
        super(DeepLift, self).__init__(net, **kwargs)
        self.baseline_fn = baseline_fn
        self.backward_hooks = []
        self.apply_rules()

    def calculate_reference_activations(self, reference):
        """
        Calculates and stores the activations from the reference input
        :param reference: the reference input
        :return: /
        """
        hooks = []

        def fw_hook(module, inp, outp):
            if not hasattr(module, 'reference_activation'):
                module.reference_inp = [inp[0]]
                module.reference_activation = [outp]
            else:
                module.reference_inp.append(inp[0])
                module.reference_activation.append(outp)

        for layer in self.net.modules():
            hooks.append(layer.register_forward_hook(fw_hook))

        self.net(reference)
        for hook in hooks:
            hook.remove()

    def calculate_diff_from_reference(self, in_values):
        """
        Calculates and stores the difference between the activation for the reference input
        and the activation for the original input.
        :param in_values: Input
        :return:  /
        """
        hooks = []

        def fw_hook(module, inp, outp):
            if not hasattr(module, 'delta_inp'):
                module.delta_inp = [inp[0] - module.reference_inp[0]]
                module.delta_outp = [outp - module.reference_activation[0]]
            else:
                module.delta_inp.append(inp[0] - module.reference_inp[len(module.delta_inp)])
                module.delta_outp.append(outp - module.reference_activation[len(module.delta_outp)])

        for layer in self.net.modules():
            hooks.append(layer.register_forward_hook(fw_hook))

        out = self.net(in_values)
        for hook in hooks:
            hook.remove()
        return out

    def apply_rules(self):
        """
        Assign the DeepLift rules to the correct layers.
        :return:
        """
        def linear_rule(module, out_relevance, in_relevance):
            delta_outp = module.delta_outp.pop()
            delta_inp = module.delta_inp.pop()
            delta_outp_pos = (delta_outp > 0) * delta_outp
            delta_outp_neg = (delta_outp < 0) * delta_outp
            ind_delta_inp_pos = (delta_inp > 0)
            ind_delta_inp_neg = (delta_inp < 0)
            ind_delta_inp_zero = (delta_inp == 0)  # TODO: use a epsilon here?

            pos_in_relevance = delta_outp_pos * in_relevance[0]
            neg_in_relevance = delta_outp_neg * in_relevance[0]

            if not isinstance(module, nn.Linear) and not isinstance(module, nn.Conv2d):
                return out_relevance

            if isinstance(module, nn.Linear):
                weight_T = module.weight.T
                weight_T_pos = (weight_T > 0) * weight_T
                weight_T_neg = (weight_T < 0) * weight_T
                out = F.linear(pos_in_relevance, weight_T_pos) * ind_delta_inp_pos \
                    + F.linear(pos_in_relevance, weight_T_neg) * ind_delta_inp_neg \
                    + F.linear(neg_in_relevance, weight_T_pos) * ind_delta_inp_neg \
                    + F.linear(neg_in_relevance, weight_T_neg) * ind_delta_inp_pos \
                    + F.linear(0.5*(pos_in_relevance + neg_in_relevance), weight_T) * ind_delta_inp_zero
                out_relevance = (out_relevance[0], out, out_relevance[2])
                return out_relevance

            elif isinstance(module, nn.Conv2d):  # TODO: support other Conv layers?
                weight = module.weight
                weight_pos = (weight > 0) * weight
                weight_neg = (weight < 0) * weight

                conv_params = copy.deepcopy(module.state_dict())
                conv_params['bias'] = torch.zeros(module.in_channels)

                pos_conv_params = copy.deepcopy(conv_params)
                pos_conv_params['weight'] = weight_pos

                neg_conv_params = copy.deepcopy(conv_params)
                neg_conv_params['weight'] = weight_neg

                params = {'stride': module.stride, 'kernel_size': module.kernel_size, 'padding': module.padding,
                          'output_padding': module.output_padding, 'padding_mode': module.padding_mode,
                          'dilation': module.dilation, 'groups': module.groups, 'in_channels': module.out_channels,
                          'out_channels': module.in_channels}

                pos_transpose_conv = nn.ConvTranspose2d(**params)
                pos_transpose_conv.load_state_dict(pos_conv_params)

                neg_transpose_conv = nn.ConvTranspose2d(**params)
                neg_transpose_conv.load_state_dict(neg_conv_params)

                transpose_conv = nn.ConvTranspose2d(**params)
                transpose_conv.load_state_dict(conv_params)

                out_size = ind_delta_inp_pos.shape
                out = pos_transpose_conv(pos_in_relevance, output_size=out_size) * ind_delta_inp_pos \
                    + neg_transpose_conv(pos_in_relevance, output_size=out_size) * ind_delta_inp_neg \
                    + pos_transpose_conv(neg_in_relevance, output_size=out_size) * ind_delta_inp_neg \
                    + neg_transpose_conv(neg_in_relevance, output_size=out_size) * ind_delta_inp_pos \
                    + transpose_conv(0.5*(pos_in_relevance + neg_in_relevance), output_size=out_size) * ind_delta_inp_zero

                out_relevance = (out, *out_relevance[1:])  # First value is the gradient of the input
                return out_relevance
            else:
                return out_relevance

        def rescale_rule(module, out_relevance, in_relevance):
            zero_threshold = 1e-7
            delta_outp = module.delta_outp.pop()
            delta_inp = module.delta_inp.pop()

            multiplier = safe_divide(delta_outp, delta_inp)

            far_zero = (delta_inp.abs() > zero_threshold) * multiplier
            near_zero = (delta_inp.abs() <= zero_threshold) * 0.01  # TODO: is this necessary

            return in_relevance[0] * (far_zero + near_zero),

        def reveal_cancel(module, out_relevance, in_relevance):

            f = F.relu
            if isinstance(module, nn.Sigmoid):
                f = F.sigmoid
            if isinstance(module, nn.Tanh):
                f = F.tanh

            _ = module.delta_outp.pop()
            delta_inp = module.delta_inp.pop()
            delta_inp_pos = (delta_inp >= 0) * delta_inp
            delta_inp_neg = (delta_inp < 0) * delta_inp
            ref_inp = module.reference_inp.pop()
            _ = module.reference_activation.pop()

            delta_outp_pos = 1/2 * ((f(ref_inp + delta_inp_pos) - f(ref_inp)) +
                                    (f(ref_inp + delta_inp_pos + delta_inp_neg) - f(ref_inp + delta_inp_neg)))

            delta_outp_neg = 1/2 * ((f(ref_inp + delta_inp_neg) - f(ref_inp)) +
                                    (f(ref_inp + delta_inp_pos + delta_inp_neg) - f(ref_inp + delta_inp_pos)))

            ind_pos_in_relevance = (out_relevance[0] >= 0)
            ind_neg_in_relevance = (out_relevance[0] < 0)

            return ind_pos_in_relevance * safe_divide(delta_outp_pos, delta_inp_pos) + \
                   ind_neg_in_relevance * safe_divide(delta_outp_neg, delta_inp_neg),

        for layer in self.net.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                self.backward_hooks.append(layer.register_backward_hook(linear_rule))
            elif isinstance(layer, nn.ReLU) or isinstance(layer, nn.Tanh) or isinstance(layer, nn.Sigmoid):
                self.backward_hooks.append(layer.register_backward_hook(rescale_rule))

    def _explain(self, in_values: torch.Tensor, labels: torch.Tensor, **kwargs) -> np.ndarray:

        self.calculate_reference_activations(self.baseline_fn.mask(in_values))
        in_values.requires_grad_(True)
        out_values = self.calculate_diff_from_reference(in_values)
        grad_out = torch.scatter(torch.zeros_like(out_values), 1, labels, torch.gather(out_values, 1, labels))
        grad = torch.autograd.grad(out_values, in_values, grad_out)[0]

        return grad.detach().cpu().numpy()


class MarginalWinningProbability(SaliencyMethod):
    def __init__(self, net: nn.Module, **kwargs):
        super(MarginalWinningProbability, self).__init__(net)

        self.activations = []
        self.maps = []

        self.forward_hooks = []
        self.backward_hooks = []

    def explain(self, in_values: torch.tensor, labels: torch.Tensor, **kwargs) -> np.ndarray:
        pass
