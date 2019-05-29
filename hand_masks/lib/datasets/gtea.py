# --------------------------------------------------------
# Dense Label Toolbox
# --------------------------------------------------------

import os
from datasets.imdb import imdb
import glob
import numpy as np
import cPickle
from dense_label.config import cfg
import cv2
import multiprocessing

class gtea(imdb):
    def __init__(self, image_set, year, devkit_path=None):
        imdb.__init__(self, 'gtea_' + year + '_' + image_set)
        self._year = year
        # image_set can be train/test
        self._image_set = image_set
        self._devkit_path = self._get_default_path() if devkit_path is None \
                            else devkit_path
        self._data_path = self._devkit_path

        self._classes = ('__background__', # always index 0
                         'hand')
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))

        self._image_ext = '.jpg'
        self._label_ext = '.png'

        # image / label share the same index
        self._image_index = self._load_image_set_index()

        # Default to roidb handler
        self._roidb_handler = self.gt_roidb
        self._comp_id = 'gtea_test'

        # GTEA specific config options
        self.config = {'label_thrs'     : 0.1, 
                       'eval_thrs'      : 40}

        assert os.path.exists(self._devkit_path), \
                'GTEA path does not exist: {}'.format(self._devkit_path)

    # a set of helper functions for loading images / labels in the dataset
    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        image_path = os.path.join(self._data_path, self._image_set, 
                                    'Images', index + self._image_ext)
        assert os.path.exists(image_path), \
                'Image path does not exist: {}'.format(image_path)
        return image_path

    def label_path_at(self, i):
        """
        Return the absolute path to label i in the label sequence.
        """
        return self.label_path_from_index(self._image_index[i])

    def label_path_from_index(self, index):
        """
        Construct a label path from the image's "index" identifier.
        """
        label_path = os.path.join(self._data_path, self._image_set, 
                                    'Masks', index + self._label_ext)
        assert os.path.exists(label_path), \
                'Label path does not exist: {}'.format(label_path)
        return label_path

    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        # Example path to image set folder:
        # self._devkit_path
        image_set_folder = os.path.join(self._data_path, self._image_set, 'Images')
        label_set_folder = os.path.join(self._data_path, self._image_set, 'Masks')

        assert os.path.isdir(image_set_folder), \
                'Path does not exist: {}'.format(image_set_folder)
        assert os.path.isdir(label_set_folder), \
                'Path does not exist: {}'.format(label_set_folder)

        # search all subfolders
        image_files = glob.glob(os.path.join(image_set_folder, '*/*' + self._image_ext))

        # loop over all image files to get their unique "index" identifier
        image_index = []
        for image_file in image_files:
            file_parts = image_file.split('/')
            file_name = file_parts[-1].split('.')[0]
            file_index = os.path.join(file_parts[-2], file_name)
            # only add image_index if both image/label exist
            if os.path.exists(os.path.join(label_set_folder, 
                                    file_index + self._label_ext)):
                image_index.append(file_index)

        return image_index

    def _get_default_path(self):
        """
        Return the default path where GTEA hand mask is expected to be installed.
        """
        return os.path.join(cfg.DATA_DIR, 'GTEA_GAZE_PLUS')

    def output_path_at(self, i, output_dir):

        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        
        filename = os.path.join(output_dir, self._image_index[i] + '.png')

        return filename

    def gt_roidb(self):
        """
        Return the database of ground-truth edge annotations.
        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_labeldb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} gt roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        gt_roidb = [self._load_gtea_annotation(index)
                    for index in self.image_index]
        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt roidb to {}'.format(cache_file)

        return gt_roidb

    def _load_gtea_annotation(self, index):
        """
        Load hand masks
        """
        label_filename = self.label_path_from_index(index)

        assert os.path.exists(label_filename), \
               'GTEA label not found at: {}'.format(filename)

        label = cv2.imread(label_filename, cv2.IMREAD_GRAYSCALE)

        # we assume the image has the same shape as its labels
        return {'label'  : label, 
                'width'  : label.shape[1],
                'height' : label.shape[0]}

    def process_label(self, label):
        ''' 
        Threshold the edge map for training
        '''
        # binary threshold of the labels
        label = label.astype(np.float32)
        label /= np.float32(255)

        label[np.logical_and(label>0.0, label<self.config['label_thrs'])] = 0.0

        # mark all hands as 1
        label[label>=self.config['label_thrs']] = 1.0

        return label

    def process_output(self, score):
        # for hand segmentation, simply return the raw scores
        return score

    def write_output_at(self, label, i, output_dir):
        '''
        Write all edges into png files 
        '''
        # check output folder

        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        
        filename = os.path.join(output_dir, self._image_index[i] + '.png')
        curr_folder,_ = os.path.split(filename)

        if not os.path.isdir(curr_folder):
            os.mkdir(curr_folder)

        # process label data
        label[label<0.0] = 0
        label[label>1.0] = 1.0
        label = (255.0*label).astype(np.uint8)

        # now we can write png files (in 8bit png)
        cv2.imwrite(filename, label)
        
    def _do_python_eval(self, output_dir = 'output'):
        '''
        Simple Evaluation Code
        '''
        pass

        # call python evaluation code (after NMS)
        # NOTE: by default, this will occupy all cores
        #gt_folder = os.path.join(self._data_path, self._image_set, 'Masks')
        #max_threads = multiprocessing.cpu_count()
        #f1, ap = segEvalDir(output_dir, gt_folder, self.config['eval_thrs'], numThreads=max_threads)

        # AP is not going to be accurate with less thrs
        #print('ODS = {:.4f}, AP = {:.4f}'.format(f1, ap))
        #print('-- Thanks, Yin Li')
        #print('--------------------------------------------------------------')


    def evaluate_labeling(self, output_dir):
        self._do_python_eval(output_dir)

    def competition_mode(self, on):
        pass
