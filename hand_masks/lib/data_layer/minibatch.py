# --------------------------------------------------------
# Dense Label Toolbox
# --------------------------------------------------------

"""Compute minibatch blobs for training a dense label network."""

import numpy as np
import cv2
from dense_label.config import cfg
from utils.blob import prep_im_for_blob, prep_label_for_blob, im_list_to_blob, label_list_to_blob
from utils.data_aug import random_color, random_flip, random_rot, random_crop, random_scale

DEBUG = False

def get_minibatch(roidb):
    """Given a roidb, construct a minibatch sampled from it."""
    num_images = len(roidb)
    # Sample random scales to use for each image in this batch
    random_scale_inds = np.random.randint(0, high=len(cfg.TRAIN.SCALES),
                                    size=num_images)

    # Randomly flip the images
    if cfg.TRAIN.USE_FLIPPED:
        random_flip_inds = np.random.randint(0, high=2, size=num_images)
    else:
        random_flip_inds = np.zeros(num_images)
    random_flip_inds = random_flip_inds.astype(dtype=np.bool)

    # Randomly rotate the images
    if cfg.TRAIN.ROT_AUG:
        random_rot_inds = np.random.randint(0, high=len(cfg.TRAIN.ROT_ANGLE), size=num_images)
    else:
        random_rot_inds = np.zeros(num_images)

    assert(cfg.TRAIN.IMS_PER_BATCH % num_images == 0), \
        'num_images ({}) must divide BATCH_SIZE ({})'. \
        format(num_images, cfg.TRAIN.BATCH_SIZE)

    # Get the input image / label blob, formatted for caffe
    im_blob, label_blob, im_scales = _get_image_label_blob(roidb, 
                                            random_flip_inds, random_scale_inds, random_rot_inds)

    blobs = {'data': im_blob, 'label' : label_blob}

    if DEBUG:
        print random_flip_inds, random_rot_inds
        print np.min(label_blob), np.max(label_blob)
        _vis_minibatch(im_blob, label_blob)
    
    return blobs

def _get_image_label_blob(roidb, flip_inds, scale_inds, rot_inds):
    """Builds an input blob from the images in the roidb at the specified
    scales.
    """
    num_images = len(roidb)
    processed_ims = []
    processed_labels = []
    im_scales = []
    for i in xrange(num_images):

        im = cv2.imread(roidb[i]['image'].encode('utf-8'))
        # we save all annotations in memory
        # always convert label to float32 for caffe
        label = roidb[i]['label'].astype(np.float32)

        # the sequence of data augmentation is important
        # scale -> flip -> rotation -> crop -> color -> re-scale

        # data augmentation: scale 
        target_size = cfg.TRAIN.SCALES[scale_inds[i]]
        if cfg.TRAIN.ROT_AUG and len(cfg.TRAIN.SCALES) > 1:
            im, label = random_scale(im, label, target_size, cfg.TRAIN.MAX_SIZE)

        # data augmentation: flip
        if cfg.TRAIN.USE_FLIPPED and flip_inds[i]:
            im, label = random_flip(im, label, flip_inds[i])

        # data augmentation: rotation
        if cfg.TRAIN.ROT_AUG:
            im, label = random_rot(im, label, cfg.TRAIN.ROT_ANGLE[rot_inds[i]])

        # data augmentation: crop
        if cfg.TRAIN.CROP_AUG:
            im, label = random_crop(im, label, cfg.TRAIN.CROP_SIZE)
        
        # data augmentation: color jittering
        if cfg.TRAIN.COLOR_AUG:
            im = random_color(im, cfg.TRAIN.COLOR_VAR)

        # data augmentation: re-scale 
        im, im_scale = prep_im_for_blob(im, cfg.PIXEL_MEANS, target_size,
                                        cfg.TRAIN.MAX_SIZE)
        label = prep_label_for_blob(label, im_scale)
        
        im_scales.append(im_scale)
        processed_ims.append(im)
        processed_labels.append(label)

    # Create a blob to hold the input images
    im_blob = im_list_to_blob(processed_ims)
    label_blob = label_list_to_blob(processed_labels)

    return im_blob, label_blob, im_scales

def _vis_minibatch(im_blob, label_blob):
    """Visualize a mini-batch for debugging."""
    import matplotlib.pyplot as plt

    for i in xrange(im_blob.shape[0]):
        im = im_blob[i, :, :, :].transpose((1, 2, 0)).copy()
        im += cfg.PIXEL_MEANS
        im = im[:, :, (2, 1, 0)]
        im = im.astype(np.uint8)

        label = np.squeeze(label_blob[i, :, :])
        label[label<=0.0] = 0
        
        f, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(im)
        ax2.imshow(label, cmap=plt.get_cmap('gray'))

        plt.show()
