# --------------------------------------------------------
# Dense Label Toolbox
# --------------------------------------------------------

import numpy as np
import cv2
import colorsys

def color_to_label(color_map, inv_color_dict):
    """ 
    Convert a color map to a label map 
    inv_color_dict is a hash table that maps color -> label
    """
    label_size = (color_map.shape[0], color_map.shape[1])
    color_map_uint32 = color_map.astype(np.uint32)
    hash_color_map = color_map_uint32[0] + 256 * color_map_uint32[1] + 65536 * color_map_uint32[2]
    label = inv_color_dict[hash_color_map.ravel()].reshape(label_size)

    return label

def label_to_color(label, color_dict):
    """ 
    Convert a label map to a colored map 
    color_dict is a hash table that maps label -> color
    """
    image_size = (label.shape[0], label.shape[1], 3)
    color_map = color_dict[label.ravel()].reshape(image_size)

    return color_map

def get_distinct_colors(num_colors):
    """
    Generate perceptual distinct colors by sampling HSV space
    """
    colors={}
    for i in np.arange(0., 360., 360. / num_colors):
        hue = i/360.
        lightness = (50 + np.random.rand() * 10)/100.
        saturation = (90 + np.random.rand() * 10)/100.
        colors[i] = colorsys.hls_to_rgb(hue, lightness, saturation)

    return colors

