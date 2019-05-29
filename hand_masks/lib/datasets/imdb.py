# --------------------------------------------------------
# Dense Label Toolbox
# --------------------------------------------------------

import os
import os.path as osp
import numpy as np
from dense_label.config import cfg

class imdb(object):
    """Image database."""

    def __init__(self, name):
        self._name = name
        self._num_classes = 0
        self._classes = []
        self._image_index = []
        self._roidb = None
        # _roidb_handler decides which function to use for loading labels
        self._roidb_handler = self.default_roidb
        # Use this dict for storing dataset specific config options
        self.config = {}

    @property
    def name(self):
        return self._name

    @property
    def num_classes(self):
        return len(self._classes)

    @property
    def classes(self):
        return self._classes

    @property
    def image_index(self):
        return self._image_index

    @property
    def roidb_handler(self):
        return self._roidb_handler

    @property
    def conifg(self):
        return self.conifg

    @roidb_handler.setter
    def roidb_handler(self, val):
        self._roidb_handler = val

    @property
    def roidb(self):
        # A roidb is a list of dictionaries, each with the following keys:
        #   image (path)
        #   label (data)
        if self._roidb is not None:
            return self._roidb
        self._roidb = self.roidb_handler()
        return self._roidb

    @property
    def cache_path(self):
        cache_path = osp.abspath(osp.join(cfg.DATA_DIR, 'cache'))
        if not os.path.exists(cache_path):
            os.makedirs(cache_path)
        return cache_path

    @property
    def num_images(self):
        return len(self.image_index)

    def image_path_at(self, i):
        raise NotImplementedError

    def label_path_at(self, i):
        raise NotImplementedError

    def default_roidb(self):
        raise NotImplementedError

    def output_path_at(self, i, output_dir):
        raise NotImplementedError

    def write_output_at(self, label, i, output_dir):
        raise NotImplementedError

    def process_label(self, label):
        '''
        Dataset specific process of labels
        '''
        raise NotImplementedError

    def process_output(self, score):
        '''
        Dataset specific process of output scores
        '''
        raise NotImplementedError

    def evaluate_labeling(self, output_dir=None):
        """
        Results should already be buffered to output_dir
        """
        raise NotImplementedError

    def _get_widths(self):
        raise NotImplementedError

    def create_roidb_from_label_list(self, label_list):
        assert len(label_list) == self.num_images, \
                'Number of labeled images must match number of ground-truth images'
        roidb = []
        for i in xrange(self.num_images):
            
            label = label_list[i]
            roidb.append({
                'label' : label,
            })
        return roidb

    def competition_mode(self, on):
        """Turn competition mode on or off."""
        raise NotImplementedError
