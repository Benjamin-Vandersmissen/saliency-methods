import torch
from PIL import Image
from matplotlib import cm
import numpy as np


def overlay(image, saliency_map, colormap=cm.inferno):

    if len(saliency_map.shape) == 3:  # Only allow pixel-level saliency maps
        saliency_map = np.mean(saliency_map, axis=0)
    heatmap = np.uint8(255 * colormap(saliency_map))

    if isinstance(image, torch.Tensor):
        image = image.cpu().detach().numpy()
        image = np.transpose(np.uint8(255 * image), (1, 2, 0))

    assert (heatmap.shape[:2] == image.shape[:2])

    image = Image.fromarray(image, mode='RGB')
    heatmap = Image.fromarray(heatmap).convert('RGB')
    heatmap.putalpha(200)  # Arbitrary transparency value, could be changed later.
    image.paste(heatmap, (0, 0), heatmap)
    return image


def save_overlay(image, saliency_map, fname, colormap=cm.inferno):
    """
    Overlay the saliency map on the image (with transparancy) and save it to a file.
    :param image: The image from which the saliency map is generated
    :param saliency_map: A saliency map
    :param fname: The filename without extension.
    :param colormap: The matplotlib colormap to visualise the saliency map with.
    :return: /
    """
    overlay(image, saliency_map, colormap).save(fname+".png")


def save(saliency_map, fname, colormap=cm.inferno):
    """
    Save a visualisation of the saliency map and save it to a file.
    :param saliency_map: A saliency map
    :param fname: The filename without extension.
    :param colormap: The matplotlib colormap to visualise the saliency map with.
    :return: /
    """
    if len(saliency_map.shape) == 2:
        # Only pixel level data
        heatmap = 255*colormap(saliency_map)
        heatmap = np.uint8(heatmap)
        Image.fromarray(heatmap).convert('RGB').save(fname + ".png")
    elif len(saliency_map.shape) == 3:
        heatmap = np.transpose(np.uint8(255*saliency_map), (1, 2, 0))
        Image.fromarray(heatmap).save(fname + ".png")
