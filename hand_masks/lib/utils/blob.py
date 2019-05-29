# --------------------------------------------------------
# Dense Label Toolbox
# --------------------------------------------------------

"""Blob helper functions."""

import numpy as np
import cv2
from utils.label import color_to_label

def im_list_to_blob(ims):
    """Convert a list of images into a network input.

    Assumes images are already prepared (means subtracted, BGR order, ...).
    """
    max_shape = np.array([im.shape for im in ims]).max(axis=0)
    num_images = len(ims)
    blob = np.zeros((num_images, max_shape[0], max_shape[1], 3),
                    dtype=np.float32)
    for i in xrange(num_images):
        im = ims[i]
        blob[i, 0:im.shape[0], 0:im.shape[1], :] = im
    # Move channels (axis 3) to axis 1
    # Axis order will become: (batch elem, channel, height, width)
    channel_swap = (0, 3, 1, 2)
    blob = blob.transpose(channel_swap)
    return blob

def prep_im_for_blob(im, pixel_means, target_size, max_size):
    """Mean subtract and scale an image for use in a blob."""
    im = im.astype(np.float32, copy=False)
    im -= pixel_means
    im_shape = im.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(target_size) / float(im_size_min)
    # Prevent the biggest axis from being more than MAX_SIZE
    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)
    im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale,
                    interpolation=cv2.INTER_CUBIC)

    return im, im_scale

def prep_label_for_blob(im_label, im_scale):
    """Convert a label map for use in a blob."""
    label = cv2.resize(im_label, None, None, fx=im_scale, fy=im_scale,
                    interpolation=cv2.INTER_NEAREST)
    label = label.astype(np.float32)

    return label

def label_list_to_blob(labels):
    """Convert a list of labels into a network input.

    Assumes -1 as the ignore label.
    """
    max_shape = np.array([label.shape for label in labels]).max(axis=0)
    num_images = len(labels)
    blob = np.zeros((num_images, max_shape[0], max_shape[1]),
                    dtype=np.float32) - 1
    for i in xrange(num_images):
        label = labels[i]
        blob[i, 0:label.shape[0], 0:label.shape[1]] = label
    # Axis order now become: (batch elem, height, width)
    return blob