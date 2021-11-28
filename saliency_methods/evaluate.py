import math
import numpy as np
import torch
from torch.nn.functional import softmax
from .utils import importance
from .mask import BlurredMask, MeanMask
import matplotlib.pyplot as plt


def intersection_over_union(saliency, masks):
    """
    :param saliency: a 4D numpy.ndarray representing a batch of saliency maps
    :param masks: a 3D numpy.ndarray representing a batch of Ground Truth masks
    :return: The intersection over union score for each saliency map.
    """

    nonzero_saliency = saliency.copy()
    nonzero_saliency[nonzero_saliency > 0] = 1
    nonzero_saliency = np.max(nonzero_saliency, axis=1)  # We want to see which pixel has relevance, not pixel+channel.

    intersection = np.count_nonzero(np.logical_and(nonzero_saliency, masks), axis=(1, 2))
    union = np.count_nonzero(np.logical_or(nonzero_saliency, masks), axis=(1, 2))
    return intersection / union


def pointing_game(saliency, masks):
    """
    :param saliency: a 4D numpy.ndarray representing a batch of saliency maps
    :param masks: a 3D numpy.ndarray representing a batch of Ground Truth masks
    :return: The pointing game score for each saliency map.
    """

    saliency = np.max(saliency.copy(), axis=1)  # Convert to pixel level

    reshaped_saliency = saliency.reshape(saliency.shape[0], -1)
    highest_salient_indices = (np.arange(0, saliency.shape[0]),  # Add batch dimension.
                               *np.unravel_index(np.argmax(reshaped_saliency, -1), saliency.shape[1:]))

    return masks[highest_salient_indices] == 1


def relevance_mass(saliency, masks):
    """
    :param saliency: a 4D numpy.ndarray representing a batch of saliency maps
    :param masks: a 3D numpy.ndarray representing a batch of Ground Truth masks
    :return: The pointing game score for each saliency map.
    """

    saliency = np.sum(saliency, axis=1)
    relevant_saliency = saliency.copy()
    relevant_saliency[masks == 0] = 0

    total = np.sum(saliency, axis=(1, 2))
    relevant = np.sum(relevant_saliency, axis=(1, 2))

    return relevant / total


def average_drop_confidence_increase(saliency, net, images, labels):
    """
    :param saliency: a $D numpy.ndarray representing a batch of saliency maps
    :param net: The neural network trained on the images and labels
    :param images: the images to occlude, this should match on each index with the corresponding label and saliency map.
    :param labels: the correct labels, this should match on each index with the corresponding image and saliency map.
    :return: A tuple of Average Drop % and Increase in confidence values.
    """
    device = next(net.parameters()).device

    images = images.to(device)
    labels = labels.view((images.shape[0], 1)).to(device)
    masked_images = images.clone() * torch.Tensor(saliency != 0, device=device)

    score = torch.gather(softmax(net(images), dim=1), 1, labels)
    new_score = torch.gather(softmax(net(masked_images), dim=1), 1, labels)

    drop = (score-new_score)/score
    confidence_increase = new_score > score
    return drop, confidence_increase


def deletion(saliency, net, images, labels, nr_steps=100, batch_size=-1):
    """
    :param saliency: a $D numpy.ndarray representing a batch of saliency maps
    :param net: The neural network trained on the images and labels
    :param images: the images to occlude, this should match on each index with the corresponding label and saliency map.
    :param labels: the correct labels, this should match on each index with the corresponding image and saliency map.
    :param nr_steps: In how many steps do we delete the image, more steps is slower but generates a more accurate curve.
    :param batch_size: How many images to process at once, (-1 = all images at once).
    :return: An array representing the scores of the deleted images at each step in the process .
    """

    if batch_size == -1:
        batch_size = images.shape[0]

    device = next(net.parameters()).device
    data_size = images.shape[0]

    labels = labels.view((data_size, 1)).to(device)

    saliency_importance = torch.tensor(importance(saliency), device=device)
    step = math.ceil(saliency[0, 0].size / nr_steps)

    deletion_scores = np.ndarray((data_size, nr_steps + 1))

    with torch.no_grad():
        for i in range(math.ceil(data_size/batch_size)):
            batch_index = batch_size*i
            batch_images = images[batch_index:batch_index+batch_size]
            batch_labels = labels[batch_index:batch_index+batch_size]
            for j in range(nr_steps+1):
                scores = torch.gather(softmax(net(batch_images), 1), 1, batch_labels).squeeze()
                deletion_scores[batch_index:batch_index+batch_size, j] = scores.detach().cpu().numpy()

                if j == nr_steps:
                    break

                indices = saliency_importance[:, :, j*step:(j+1)*step]

                for k in range(batch_size):
                    batch_images[k, :, indices[k, 0], indices[k, 1]] = 0

    return deletion_scores


def insertion(saliency, net, images, labels, nr_steps=100, batch_size=-1):
    """
    :param saliency: a $D numpy.ndarray representing a batch of saliency maps
    :param net: The neural network trained on the images and labels
    :param images: the images to occlude, this should match on each index with the corresponding label and saliency map.
    :param labels: the correct labels, this should match on each index with the corresponding image and saliency map.
    :param nr_steps: In how many steps do we delete the image, more steps is slower but generates a more accurate curve.
    :param batch_size: How many images to process at once, (-1 = all images at once).
    :return: An array representing the scores of the deleted images at each step in the process .
    """

    if batch_size == -1:
        batch_size = images.shape[0]

    device = next(net.parameters()).device
    data_size = images.shape[0]

    labels = labels.view((data_size, 1)).to(device)

    saliency_importance = torch.tensor(importance(saliency), device=device)
    step = math.ceil(saliency[0, 0].size / nr_steps)

    insertion_scores = np.ndarray((data_size, nr_steps + 1))
    blur = BlurredMask()

    with torch.no_grad():
        for i in range(math.ceil(data_size/batch_size)):
            batch_index = batch_size*i
            batch_images = images[batch_index:batch_index+batch_size]
            batch_blurred_images = blur.mask(batch_images)
            batch_labels = labels[batch_index:batch_index+batch_size]
            for j in range(nr_steps+1):
                scores = torch.gather(softmax(net(batch_blurred_images), 1), 1, batch_labels).squeeze()
                insertion_scores[batch_index:batch_index+batch_size, j] = scores.detach().cpu().numpy()

                if j == nr_steps:
                    break

                indices = saliency_importance[:, :, j*step:(j+1)*step]

                for k in range(batch_size):
                    batch_blurred_images[k, :, indices[k, 0], indices[k, 1]] = \
                        batch_images[k, :, indices[k, 0], indices[k, 1]]

    return insertion_scores
