import torch
import torch.nn as nn


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

    3D-numpy.ndarray
        A saliency map for the first image in the batch.

    """
    device = next(net.parameters()).device

    dummy_value = torch.zeros(shape).to(device)
    layers, handles = [], []

    func = lambda module, input, output: layers.append(module)

    for name, m in net.named_modules():
        if len(list(m.named_modules())) == 1:  # Only ever have the singular layers, no sub networks etc
            handles.append(m.register_forward_hook(func))

    net.forward(dummy_value)
    for handle in handles:
        handle.remove()

    return layers
