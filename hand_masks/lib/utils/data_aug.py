# --------------------------------------------------------
# Dense Label Toolbox
# --------------------------------------------------------

import numpy as np
import cv2
from dense_label.config import cfg

# This file implements a set of data augmentations for 
# image and its dense label, including rotation, flipping, 
# color jitter and cropping

def random_scale(im, label, target_size, max_size):
    """
    scale image into target size(capped by max_size)
    """
    assert (im.shape[0] == label.shape[0]) and (im.shape[1] == label.shape[1])

    im_shape = im.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(target_size) / float(im_size_min)
    # Prevent the biggest axis from being more than MAX_SIZE
    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)

    im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale,
                    interpolation=cv2.INTER_CUBIC)

    label = cv2.resize(label, None, None, fx=im_scale, fy=im_scale,
                    interpolation=cv2.INTER_NEAREST)

    return im, label

def random_color(im, color_var):
    assert im.dtype == np.uint8
    """
    Add a random color jitter over three channels.
    The function is implemented using LUT for efficiency
    """
    im_color = im.copy()
    rand_coeff = 2.0*color_var*(np.random.rand(1,3)-0.5) + 1.0
    pixel_value = np.arange(256).astype(np.float32)
    for c in xrange(3):
        lut_table = pixel_value * rand_coeff[0, c]
        lut_table = np.minimum(lut_table, 255.0)
        lut_table = np.round(lut_table).astype(np.uint8)
        im_color[:,:,c] = cv2.LUT(im[:,:,c], lut_table)

    return im_color

def random_flip(im, label, flip=None):
    """
    Flip the image and its label horizontally at random
    """
    if flip is None:
        flip = np.random.randint(0, high=2, size=1)
    if flip:
        im = im[:, ::-1, :]
        label = label[:, ::-1]
    return im, label

def random_rot(im, label, rot_angle=None):
    """
    Rotate the image and its label at random
    """
    # rot angle is defined counter clock-wise
    if rot_angle is None:
        rot_idx = np.random.randint(0, high=len(cfg.TRAIN.ROT_ANGLE), size=1)
        rot_angle = float(cfg.TRAIN.ROT_ANGLE[rot_idx])

    # corner case where we can do it fast!
    if np.abs(rot_angle + 90) < 1:
        im = cv2.flip(cv2.transpose(im), 1)
        label = cv2.flip(cv2.transpose(label),1)
    elif np.abs(rot_angle+180) < 1:
        im = im[::-1, :, :]
        label = label[::-1, :]
    elif np.abs(rot_angle+270) < 1:
        im = cv2.flip(cv2.transpose(im), 0)
        label = cv2.flip(cv2.transpose(label), 0)
    else:
        h, w = im.shape[0], im.shape[1]
        side_long = float(np.max(im.shape[0:2]))
        side_short = float(np.min(im.shape[0:2]))

        # since the solutions for angle, -angle and pi-angle are all the same,
        # it suffices to look at the first quadrant and the absolute values of sin,cos:
        sin_a = np.abs(np.sin(np.pi * rot_angle / 180))
        cos_a = np.abs(np.cos(np.pi * rot_angle / 180))

        if (side_short <= 2.0*sin_a*cos_a*side_long):
            # half constrained case: two crop corners touch the longer side,
            # the other two corners are on the mid-line parallel to the longer line
            x = 0.5*side_short
            if w>=h:
                wr, hr = x/sin_a, x/cos_a
            else:
                wr, hr = x/cos_a, x/sin_a
        else:
            # fully constrained case: crop touches all 4 sides
            cos_2a = cos_a*cos_a - sin_a * sin_a
            wr = (w*cos_a - h*sin_a)/cos_2a
            hr = (h*cos_a - w*sin_a)/cos_2a

        rot_mat = cv2.getRotationMatrix2D((w/2.0, h/2.0), rot_angle, 1.0)
        rot_mat[0,2] += (wr - w)/2.0
        rot_mat[1,2] += (hr - h)/2.0

        im = cv2.warpAffine(im, rot_mat, (int(wr), int(hr)), flags=cv2.INTER_CUBIC)
        label = cv2.warpAffine(label, rot_mat, (int(wr), int(hr)), flags=cv2.INTER_NEAREST)

    return im, label

def random_crop(im, label, crop_size, crop_w=None, crop_h=None):
    """
    Crop the image and its label at random
    """
    # crop_size in (width, height)
    width, height = im.shape[1], im.shape[0]
    if (crop_w is None) and (crop_h is None): 
        crop_w = np.random.randint(0, high = width - crop_size[0], size=1)[0]
        crop_h = np.random.randint(0, high = height - crop_size[1], size=1)[0]
    im = im[crop_h:crop_h+crop_size[1], crop_w:crop_w+crop_size[0], :]
    label = label[crop_h:crop_h+crop_size[1], crop_w:crop_w+crop_size[0]]

    return im, label


# test 
if __name__ == '__main__':
    
    import sys, time
    img_file = sys.argv[1]
    label_file = sys.argv[2]

    # load image and its dense label
    tic = time.time()
    im = cv2.imread(img_file)
    label = cv2.imread(label_file)
    toc = time.time()
    print 'Loading (t=%0.3fs).'%( toc-tic )

    # color / rotation / crop
    tic = time.time()
    im_color = random_color(im, 0.15)
    
    im_rot, label_rot = random_rot(im, label, 30)
    
    im_crop, label_crop = random_crop(im, label, [640, 480])

    toc = time.time()
    
    print 'Aug (t=%0.3fs).'%( toc-tic )

    # visualization
    import matplotlib.pyplot as plt

    plt.figure(1)
    plt.subplot(1,2,1), plt.imshow(im[:,:,[2,1,0]])
    plt.subplot(1,2,2), plt.imshow(im_color[:,:,[2,1,0]])

    plt.figure(2)
    plt.subplot(1,2,1), plt.imshow(im_rot[:,:,[2,1,0]])
    plt.subplot(1,2,2), plt.imshow(label_rot[:,:,[2,1,0]])

    plt.figure(3)
    plt.subplot(1,2,1), plt.imshow(im_crop[:,:,[2,1,0]])
    plt.subplot(1,2,2), plt.imshow(label_crop[:,:,[2,1,0]])

    plt.show()

