# --------------------------------------------------------
# Dense Label Toolbox
# --------------------------------------------------------

"""Test a network on an imdb (image database)."""

from dense_label.config import cfg, get_output_dir
import argparse
from utils.timer import Timer
import numpy as np
import cv2
import caffe
from utils.blob import im_list_to_blob
from utils.label import label_to_color
import os

def _get_image_blob(im):
    """Converts an image into a network input.

    Arguments:
        im (ndarray): a color image in BGR order

    Returns:
        blob (ndarray): a data blob holding an image pyramid
        im_size (list): original image sizes (width, height)
    """
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS

    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    processed_ims = []
    im_size = []
    im_size.append((im_orig.shape[1], im_orig.shape[0]))

    for target_size in cfg.TEST.SCALES:
        im_scale = float(target_size) / float(im_size_min)
        # Prevent the biggest axis from being more than MAX_SIZE
        if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
            im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
        im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                        interpolation=cv2.INTER_CUBIC)
        processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob, np.array(im_size)

def _get_output_scores(scores, im_size):
    """Converts a network output to the same size as input image
    We always assume the output is center-aligned with input image
    Arguments:
        scores (ndarray): 1 * num_classes * h * w
        im_size (ndarray): 1 * 2 (width, height)

    Returns:
        scores : numpy array of probs
    """
    assert im_size.shape[0] == scores.shape[0] and im_size.shape[0] == 1
    
    # get scores
    cur_scores = scores.transpose((2,3,1,0))
    cur_scores = np.squeeze(cur_scores)

    # reshape the scores back to original resolution
    final_scores = cv2.resize(cur_scores, (im_size[0, 0], im_size[0, 1]), 
                        interpolation=cv2.INTER_CUBIC)

    return final_scores

def im_label(net, im):
    """Label the image

    Arguments:
        net (caffe.Net): Fast R-CNN network to use
        im (ndarray): H*W color image to test (in BGR order) 

    Returns:
        scores (ndarray): H*W*K array of label class scores
    """

    # convert image into network input format
    blobs = {'data' : None}
    blobs['data'], im_size = _get_image_blob(im)

    # reshape network inputs
    net.blobs['data'].reshape(*(blobs['data'].shape))

    # do forward
    forward_kwargs = {'data': blobs['data'].astype(np.float32, copy=False)}
    blobs_out = net.forward(**forward_kwargs)

    # rescale labelling results back to original resolution
    raw_scores = net.blobs['prob'].data
    scores = _get_output_scores(raw_scores, im_size)

    return scores.copy()

def vis_results(im, labels, color_dict):
    """Visual debugging of detections."""
    import matplotlib.pyplot as plt
    im = im[:, :, (2, 1, 0)]
    vis_label = label_to_color(labels, color_dict)
    plt.figure(1), plt.subplot(1,2,1), plt.imshow(im)
    plt.subplot(1,2,2), plt.imshow(vis_label)
    plt.imshow()

def vis_edges(im, scores):
    """Visual debugging of edge detection."""
    import matplotlib.pyplot as plt
    im = im[:, :, (2, 1, 0)]
    plt.figure(1), plt.subplot(1,2,1), plt.imshow(im)
    plt.subplot(1,2,2), plt.imshow(scores)
    plt.imshow()

def test_net(net, imdb, vis=False, benchmark=False):
    """Test a Fast R-CNN network on an image database."""
    num_images = len(imdb.image_index)
                
    output_dir = get_output_dir(imdb, net)

    # timers
    _t = {'im_label' : Timer()}

    for i in xrange(num_images):

        # check if the current image is already computed
        output_file = imdb.output_path_at(i, output_dir)

        if not os.path.exists(output_file):

            im = cv2.imread(imdb.image_path_at(i).encode('utf-8'))

            if cfg.TEST.USE_FLIPPED:
                im_flip = im[:, ::-1, :]

            _t['im_label'].tic()

            # detection on the orginal image
            scores = im_label(net, im)

            # flip the image if data augmentation is turned on during testing
            if cfg.TEST.USE_FLIPPED:
                scores_flip = im_label(net, im_flip)
                scores = 0.5*(scores + scores_flip[:, ::-1])

            # collect labels
            label = imdb.process_output(scores)
            imdb.write_output_at(label, i, output_dir)

            _t['im_label'].toc()

            if vis:
                vis_edges(im, scores)

            print 'im_label: {:d}/{:d} {:.3f}s ' \
                  .format(i + 1, num_images, _t['im_label'].average_time)
        else:
            print 'im_label: {:d}/{:d} skipped ' \
                  .format(i + 1, num_images)

    if benchmark:
        print 'Evaluating dense labeling'
        imdb.evaluate_labeling(output_dir)
