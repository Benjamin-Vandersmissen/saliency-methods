import numpy as np
import torch
from torch.nn.functional import softmax
import utils

import matplotlib.pyplot as plt

from torch.utils.data import DataLoader


def evaluate(activation_maps, test_data, net, mask_directory, method):
    testLoader = iter(DataLoader(test_data))

    images = {}
    masks = {}
    labels = {}

    for data in testLoader:
        image, label, filename = data
        label = label.squeeze().item()
        filename = filename[0]

        mask = utils.file_to_np_array(mask_directory + '/' + filename.split('/')[-2] + '/' + filename.split('/')[-1])[0]
        images[filename] = image
        masks[filename] = mask
        labels[filename] = label

    # #THRESHOLD
    activation_maps = utils.threshold_all(activation_maps, value=0)

    for typ, activations in activation_maps.items():

        if method == 'IoU' or method == 'all':
            iou = intersection_over_union(activations, masks)
            print("average IoU for {} : {}".format(typ, iou))

        if method == 'relevance_mass' or method == 'all':
            mass = relevance_mass(activations, masks)
            print("average relevance mass for {} : {} ".format(typ, mass))

        if method == 'proposed' or method == 'all':
            score = proposed(activations, masks)
            print("average proposed metric for {} : {}".format(typ, score))

        if method == 'pointing' or method == 'all':
            score = pointing_game(activations, masks)
            print("average Pointing Game score for {} : {}".format(typ, score))

        if method == 'drop_increase' or method == 'all':
            drop, increase = average_drop_confidence_increase(activations, net, images, labels)
            print("Average drop for {} : {}".format(typ, drop))
            print("Confidence increase for {} : {}".format(typ, increase))

        if method == 'relative_pixel_attribution':
            attr = relative_pixel_attribution(activations, masks)
            print("Relative Pixel Attribution for this feature using {} : {}".format(typ, attr))

        if method == 'distribution':
            distribution(activations, typ, masks)

        if method == 'deletion' or method == 'all':
            auc = deletion(activations, net, images, labels)
            print("Average AUC for deletion metric for {} : {}".format(typ, auc))

        if method == 'insertion' or method == 'all':
            auc = insertion(activations, net, images, labels)
            print("Average AUC for insertion metric for {} : {}".format(typ, auc))

        print("\b")
        # total = 0
        # for mask in masks.values():
        #     mask = mask.copy()
        #     mask = np.logical_not(mask)
        #     total += np.sum(mask)
        #
        # total /= len(masks)
        # print("FEATURE SIZE: {}".format(total))
        # print("")

def intersection_over_union(activations, masks):
    running_iou = 0
    i = 0
    for filename, activation in activations.items():
        mask = np.logical_not(masks[filename])  # 0 values in the mask indicate relevant regions

        nonzero_activation = activation.copy()
        nonzero_activation[nonzero_activation != 0] = 1

        intersection = np.count_nonzero(np.logical_and(nonzero_activation, mask))
        union = np.count_nonzero(np.logical_or(nonzero_activation, mask))

        running_iou += intersection/union
        i += 1
    return running_iou / i


def pointing_game(activations, masks):
    running_score = 0
    i = 0
    for filename, activation in activations.items():
        mask = masks[filename]
        highest_activated_index = np.unravel_index(np.argmax(activation), activation.shape)
        running_score += (mask[highest_activated_index] != 1)
        i += 1
    return running_score / i


def relevance_mass(activations, masks, negative=False):
    running_relevance_mass = 0
    i = 0
    for filename, activation in activations.items():
        mask = masks[filename]
        total = np.sum(activation[activation < 0 if negative else activation > 0])

        if np.isnan(activation).any():
            continue  # this happens sometimes with GradCAM for no apparent reason

        if total == 0:
            running_relevance_mass += 0
        else:
            activation = activation.copy()
            activation[mask == 1] = 0  # set non-ground truth to 0

            activation[activation > 0 if negative else activation < 0] = 0 # set all positive relevance to 0.

            mass = np.sum(activation) / total

            running_relevance_mass += mass

        i += 1
    return running_relevance_mass / i


def proposed(activations, masks):
    running_metric = 0
    i = 0

    for filename, activation in activations.items():
        mask = masks[filename]
        total = np.sum(activation[activation > 0])

        if total == 0:
            running_metric += 0
        else:
            activation = activation.copy()
            activation[mask == 1] = 0
            activation[activation < 0] = 0

            mass = np.sum(activation) / total

            mean = np.mean(activation[mask == 0])
            diff = np.sum(np.abs(activation[mask == 0] - mean))

            diff /= (2*np.sum(1-mask) - 2) * mean
            # metric = mass*(1-diff)

            # Only the spreading factor
            metric = (1-diff)

            running_metric += metric
        i += 1
    return running_metric/i

def relative_pixel_attribution(activations, masks):
    running_attribution = 0
    running_pixels = 0

    for filename, activation in activations.items():

        mask = np.logical_not(masks[filename])
        running_attribution += np.sum(activation[mask])
        running_pixels += np.sum(mask)

    return running_attribution / running_pixels


def distribution(activations, typ, masks):
    bla = {}

    percentage_non_gt = 0

    for i in range(1001):
        bla[i] = 0

    for filename, activation in activations.items():
        for i in range(activation.shape[0]):
            for j in range(activation.shape[1]):
                key = int(1000*round(activation[i,j], 3))
                bla[key] += 1

        percentage_non_gt += np.sum(masks[filename])

    print((activation.size * len(activations) - percentage_non_gt) / (len(activations)))

    percentage_non_gt /= activation.size * len(activations)
    distr = []
    x_trheshold = None
    for i in range(1001):
        bla[i] /= len(activations) * activation.size
        if i != 0:
            bla[i] += bla[(i-1)]

        if bla[i] > percentage_non_gt and x_trheshold is None:
            x_trheshold = i/1000  # This is an approximation, maybe better ?
        distr.append(bla[i])

    x_values = np.arange(0, 1+1/1000, 1/1000)
    plt.plot(x_values, distr)
    plt.fill_between(x_values, distr)
    plt.title("Relevance distribution for {}".format(typ))
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.xlabel("Relevance")
    plt.ylabel("Density")
    plt.axhline(y=percentage_non_gt, xmax= x_trheshold, color='r', linestyle='--')
    plt.axvline(x=x_trheshold, ymax=percentage_non_gt, color='r', linestyle='--')
    plt.show()

def average_drop_confidence_increase(activations, net, images, labels):
    running_drop = 0
    running_increase = 0
    i = 0

    max_increase = 0

    prediction_score = 0

    for filename, activation in activations.items():
        image = images[filename]
        label = labels[filename]

        masked_image = image.clone()
        masked_image[:, :, activation == 0] = 0
        score = softmax(net(image), dim=1).squeeze()[label].item()
        new_score = softmax(net(masked_image), dim=1).squeeze()[label].item()
        drop = max(0, score-new_score)/score
        confidence_increase = new_score > score

        if new_score > score:
            max_increase = max(max_increase, new_score-score)

        running_drop += drop
        running_increase += confidence_increase
        i += 1

        prediction_score += score

    print(prediction_score/i)
    print(max_increase)
    return running_drop/i, running_increase


def deletion(activations, net, images, labels, nr_steps=100):
    running_deletion_scores = np.empty((len(activations), nr_steps+1))
    k = 0

    for filename, activation in activations.items():
        image = images[filename].clone()
        label = labels[filename]

        importance = utils.importance(activation)
        step = activation.size // nr_steps

        batch = torch.zeros((nr_steps+1, *image.shape[1:]))
        batch[0] = image

        for i in range(nr_steps):
            indices = importance[:, i*step:(i+1)*step]

            if i == nr_steps - 1:
                indices = importance[:, i*step:]

            value = torch.zeros((1,3, indices.shape[1]))

            image[:, :, indices[0], indices[1]] = value
            batch[i+1] = image

        deletion_scores = net.sm_score(batch).squeeze().cpu().detach().numpy()[:, label]
        running_deletion_scores[k] = deletion_scores
        k += 1

    running_deletion_scores = running_deletion_scores.mean(axis=0)

    x_values = np.arange(0, 1+1/nr_steps, 1/nr_steps)

    fig, ax = plt.subplots()
    ax.plot(x_values, running_deletion_scores)
    plt.fill_between(x_values, running_deletion_scores)
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    plt.show()

    auc = np.trapz(running_deletion_scores, x_values)

    return auc


def insertion(activations, net, images, labels, nr_steps=100):
    running_insertion_scores = np.empty((len(activations), nr_steps+1))
    k = 0

    # from torchvision.transforms import GaussianBlur
    #
    # blur = GaussianBlur()

    for filename, activation in activations.items():
        image = images[filename]
        # blurred_image = blur(image.clone())
        blurred_image = torch.zeros_like(image)  # We use the empty image for an uninformative image.
        label = labels[filename]

        importance = utils.importance(activation)
        step = activation.size // nr_steps

        batch = torch.zeros((nr_steps+1, *image.shape[1:]))
        batch[0] = blurred_image
        for i in range(nr_steps):
            indices = importance[:, i*step:(i+1)*step]

            if i == nr_steps - 1:
                indices = importance[:, i*step:]

            value = image[:, :, indices[0], indices[1]]
            blurred_image[:, :, indices[0], indices[1]] = value

            batch[i+1] = blurred_image


        insertion_scores = net.sm_score(batch).squeeze().cpu().detach().numpy()[:, label]
        running_insertion_scores[k] = insertion_scores
        k += 1


    running_insertion_scores = running_insertion_scores.mean(axis=0)

    x_values = np.arange(0, 1+1/nr_steps, 1/nr_steps)

    plt.plot(x_values, running_insertion_scores)
    plt.fill_between(x_values, running_insertion_scores)
    plt.show()

    auc = np.trapz(running_insertion_scores, x_values)

    return auc
