import torch
import torch.nn as nn
import numpy as np


EPSILON = np.finfo(np.float32).eps


def safe_divide(a, b):
    with torch.no_grad():
        eps = np.finfo(np.float32).eps
        if isinstance(a, torch.Tensor):
            eps = torch.FloatTensor([eps])
        b[b == 0] = eps
    return a / b


def extract_layers(net: nn.Module, shape: tuple) -> list:
    """ Extract the layers from a neural network in the order they are activated in.

    Parameters
    ----------

    net : torch.nn.module
        The neural network from which to extract the layers.

    shape : 4D-tuple of shape (batch, channel, width, height)
        The shape of the input the neural network expects.

    Returns
    -------

    list of (name, module) tuples containing all modules in the net
    """
    device = next(net.parameters()).device

    dummy_value = torch.zeros(shape).to(device)
    layers, handles = [], []

    def func(name):
        return lambda module, inp, outp: layers.append((name, module))

    for name, m in net.named_modules():
        if len(list(m.named_modules())) == 1:  # Only ever have the singular layers, no sub networks etc
            handles.append(m.register_forward_hook(func(name)))

    net.forward(dummy_value)
    for handle in handles:
        handle.remove()

    return layers


def importance(heatmaps):
    shape = heatmaps.shape[2:]
    if isinstance(heatmaps, torch.Tensor):
        heatmaps = heatmaps.detach().numpy()
    heatmaps = np.sum(heatmaps, axis=1).reshape(heatmaps.shape[0], -1)  # Get relevance per pixel

    index = np.transpose(np.array(np.unravel_index(np.argsort(- heatmaps, axis=1), shape)), (1,0,2))
    return index


def auc(points):
    x_values = np.arange(0, 1+1/points.size(), 1/points.size())
    return np.trapz(points, x_values)
