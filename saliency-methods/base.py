import numpy as np
import torch
from torch import nn


#
# A base class for the saliency methods
#
class SaliencyMethod:

    def __init__(self, net: nn.Module, smoothed=False, smooth_rate=10):
        self.smoothed = smoothed
        self.smooth_rate = smooth_rate
        self.net = net

        self.net.eval()

    def calculate_mask(self, in_values: torch.Tensor, label: torch.Tensor) -> np.ndarray:
        raise NotImplementedError("A subclass needs to implement this function.")
