import math
import numpy as np
import torch
from torch.nn.functional import softmax
from .utils import importance
from .mask import BlurredMask, FullMask, UniformMask


def intersection_over_union(saliency, masks):
    """
    :param saliency: a 4D numpy.ndarray representing a batch of saliency maps
    :param masks: a 3D numpy.ndarray representing a batch of Ground Truth masks
    :return: The intersection over union score for each saliency map.
    """

    nonzero_saliency = saliency.copy()
    nonzero_saliency = nonzero_saliency > 0
    nonzero_saliency = np.max(nonzero_saliency, axis=1)  # We want to see which pixel has relevance, not pixel+channel.

    intersection = np.sum(np.logical_and(nonzero_saliency, masks), axis=(1, 2))
    union = np.sum(np.logical_or(nonzero_saliency, masks), axis=(1, 2))
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


def average_drop_confidence_increase(saliency, net, images, labels, threshold=None):
    """
    :param saliency: a 4D numpy.ndarray representing a batch of saliency maps
    :param net: The neural network trained on the images and labels
    :param images: the images to occlude, this should match on each index with the corresponding label and saliency map.
    :param labels: the correct labels, this should match on each index with the corresponding image and saliency map.
    :return: A tuple of Average Drop % and Increase in confidence values.
    """
    batch_size = images.shape[0]
    with torch.no_grad():
        device = next(net.parameters()).device
        net = net.eval()

        images = images.to(device)
        labels = labels.view((batch_size, -1)).to(device, dtype=torch.long)
        if threshold is not None:
            saliency = saliency.copy()
            saliency[saliency >= threshold] = 1
            saliency[saliency < threshold] = 0
        masked_images = images.detach().clone() * torch.tensor(saliency).to(device)

        score = torch.gather(softmax(net(images), dim=1), 1, labels).cpu()
        new_score = torch.gather(softmax(net(masked_images), dim=1), 1, labels).cpu()

        drop = torch.maximum(torch.zeros_like(score), (score-new_score))/score
        confidence_increase = new_score > score
        return drop.numpy(), confidence_increase


def _ins_del_parent(saliency, net, start, end, labels, nr_steps=100, minibatch_size=-1):
    """
    :param saliency: a 4D numpy.ndarray representing a batch of saliency maps
    :param net: The neural network trained on the images and labels
    :param images: the images to occlude, this should match on each index with the corresponding label and saliency map.
    :param labels: the correct labels, this should match on each index with the corresponding image and saliency map.
    :param nr_steps: In how many steps do we delete the image, more steps is slower but generates a more accurate curve.
    :param minibatch_size: How many images to process at once, (-1 = all images at once).
    :return: An array representing the scores of the deleted images at each step in the process .
    """

    batch_size = start.shape[0]
    if minibatch_size == -1:
        minibatch_size = batch_size

    device = next(net.parameters()).device
    net = net.eval()
    labels = labels.view((batch_size, -1)).to(device)

    #  Return the indices in the saliency maps ordered by importance.
    saliency_importance = torch.tensor(importance(saliency))
    step = math.ceil(start[0, 0].numel() / nr_steps)

    all_scores = np.ndarray((batch_size, nr_steps + 1))
    with torch.no_grad():
        for i in range(math.ceil(batch_size/minibatch_size)):
            minibatch_index = minibatch_size*i
            minibatch_end = minibatch_index + minibatch_size
            batch_start = start[minibatch_index:minibatch_end].to(device)
            batch_end = end[minibatch_index:minibatch_end].to(device)
            batch_labels = labels[minibatch_index:minibatch_end].to(device)
            batch_saliency_importance = saliency_importance[minibatch_index:minibatch_end]
            for j in range(nr_steps+1):
                scores = torch.gather(softmax(net(batch_start), 1), 1, batch_labels).reshape(-1)
                all_scores[minibatch_index:minibatch_end, j] = scores.detach().cpu().numpy()

                if j == nr_steps:  # Do one last step with the end image
                    break

                indices = batch_saliency_importance[:, :, j*step:(j+1)*step]

                for k in range(minibatch_size):
                    batch_start[k, :, indices[k, 0], indices[k, 1]] = batch_end[k, :, indices[k, 0], indices[k, 1]]
    return all_scores


def deletion(saliency, net, images, labels, nr_steps=100, minibatch_size=-1):
    """
    :param saliency: a 4D numpy.ndarray representing a batch of saliency maps
    :param net: The neural network trained on the images and labels
    :param images: the images to occlude, this should match on each index with the corresponding label and saliency map.
    :param labels: the correct labels, this should match on each index with the corresponding image and saliency map.
    :param nr_steps: In how many steps do we delete the image, more steps is slower but generates a more accurate curve.
    :param minibatch_size: How many images to process at once, (-1 = all images at once).
    :return: An array representing the scores of the deleted images at each step in the process .
    """
    return _ins_del_parent(saliency, net, images, FullMask(0).mask(images), labels, nr_steps, minibatch_size)


def insertion(saliency, net, images, labels, nr_steps=100, minibatch_size=-1, blur=BlurredMask(11, 5)):
    """
    :param saliency: a 4D numpy.ndarray representing a batch of saliency maps
    :param net: The neural network trained on the images and labels
    :param images: the images to occlude, this should match on each index with the corresponding label and saliency map.
    :param labels: the correct labels, this should match on each index with the corresponding image and saliency map.
    :param nr_steps: In how many steps do we delete the image, more steps is slower but generates a more accurate curve.
    :param minibatch_size: How many images to process at once, (-1 = all images at once).
    :param blur: Which method do we use to create an empty image.
    :return: An array representing the scores of the deleted images at each step in the process .
    """
    return _ins_del_parent(saliency, net, blur.mask(images), images, labels, nr_steps, minibatch_size)


def area_over_MoRF_curve(saliency, net, images, labels, nr_steps=100,  blur=UniformMask(), shape=(9, 9), average_over=1):
    """
    :param saliency: a 4D numpy.ndarray representing a batch of saliency maps
    :param net: The neural network trained on the images and labels
    :param images: the images to occlude, this should match on each index with the corresponding label and saliency map.
    :param labels: the correct labels, this should match on each index with the corresponding image and saliency map.
    :param nr_steps: In how many steps do we delete the image, more steps is slower but generates a more accurate curve.
    :param blur: Which method do we use to create a mask.
    :param shape: The shape of the occlusion patch
    :param average_over: How many iterations do we do for each image.
    :return: An array representing the AOPC scores .
    """
    batch_size = images.shape[0]

    device = next(net.parameters()).device
    net = net.eval()
    labels = labels.view((batch_size, -1))

    #  Return the indices in the saliency maps ordered by importance.
    saliency_importance = torch.tensor(importance(saliency))

    original = images.clone()
    original_scores = torch.gather(torch.softmax(net(original.to(device)), 1), 1, labels.to(device)).cpu()
    scores = torch.zeros((batch_size, average_over, nr_steps+1))
    for j in range(batch_size):
        for k in range(average_over):
            occluded = torch.zeros(original.shape[2:])
            idx = 0
            img = images[j].clone().to(device)
            for i in range(nr_steps+1):
                x = saliency_importance[j, 0, idx].item()
                y = saliency_importance[j, 1, idx].item()

                while occluded[x, y]:  # This pixel has already been occluded.
                    idx += 1
                    x = saliency_importance[j, 0, idx].item()
                    y = saliency_importance[j, 1, idx].item()

                # Calculate the actual positions of the window (in case of cut-off)
                left = max(x - shape[0]//2, 0)
                right = min(x + (shape[0]-1)//2, original.shape[2]-1)
                top = max(y - shape[1]//2, 0)
                bottom = min(y + (shape[1]-1)//2, original.shape[3]-1)

                occluded[left:right+1, top:bottom+1] = 1
                occluded_patch = blur.mask(original, (img.shape[0], right-left+1, bottom-top+1)).to(device)
                img[:, left:right+1, top:bottom+1] = occluded_patch
                idx += 1
                scores[j, k, i] = original_scores[j] - torch.softmax(net(img.unsqueeze(0)), 1).cpu()[0, labels[j]]

    return scores.mean(dim=1).detach().numpy()
