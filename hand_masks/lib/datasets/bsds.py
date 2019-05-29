# --------------------------------------------------------
# Dense Label Toolbox
# --------------------------------------------------------

import os
from datasets.imdb import imdb
import glob
import numpy as np
import scipy.io as sio
import cPickle
from bsds_eval import edgesEvalDir
from bsds_utils import edgeOrient, edgeNms
from dense_label.config import cfg
import cv2
import multiprocessing

class bsds(imdb):
    def __init__(self, image_set, year, devkit_path=None):
        imdb.__init__(self, 'bsds_' + year + '_' + image_set)
        self._year = year
        self._image_set = image_set
        self._devkit_path = self._get_default_path() if devkit_path is None \
                            else devkit_path
        self._data_path = os.path.join(self._devkit_path, 'data')
        self._classes = ('__background__', # always index 0
                         'edge')
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))

        self._image_ext = '.jpg'
        self._label_ext = '.mat'

        # image / label share the same index
        self._image_index = self._load_image_set_index()

        # Default to roidb handler
        self._roidb_handler = self.gt_roidb
        self._comp_id = 'bsds_test'

        # BSDS specific config options
        self.config = {'label_thrs'     : 0.45, 
                       'eval_thrs'      : 40}

        assert os.path.exists(self._devkit_path), \
                'BSDS500 path does not exist: {}'.format(self._devkit_path)
        assert os.path.exists(self._data_path), \
                'Path does not exist: {}'.format(self._data_path)

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
        image_path = os.path.join(self._data_path, 'images',
                                  self._image_set, index + self._image_ext)
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
        label_path = os.path.join(self._data_path, 'groundTruth',
                                  self._image_set, index + self._label_ext)
        assert os.path.exists(label_path), \
                'Label path does not exist: {}'.format(label_path)
        return label_path

    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        # Example path to image set folder:
        # self._devkit_path + /data/images/val
        image_set_folder = os.path.join(self._data_path, 'images',
                                      self._image_set)

        assert os.path.isdir(image_set_folder), \
                'Path does not exist: {}'.format(image_set_folder)
        image_files = glob.glob(os.path.join(image_set_folder, '*' + self._image_ext))

        # loop over all image files to get their unique "index" identifier
        image_index = []
        for image_file in image_files:
            filename = os.path.basename(image_file)
            file_index, _ = os.path.splitext(filename)
            image_index.append(file_index)

        return image_index

    def _get_default_path(self):
        """
        Return the default path where BSDS500 is expected to be installed.
        """
        return os.path.join(cfg.DATA_DIR, 'bsds')

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

        gt_roidb = [self._load_bsds_annotation(index)
                    for index in self.image_index]
        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt roidb to {}'.format(cache_file)

        return gt_roidb

    def _load_bsds_annotation(self, index):
        """
        Load edge annotation from mat file into bsds format
        """
        img_filename = self.image_path_from_index(index)
        filename = self.label_path_from_index(index)
        
        assert os.path.exists(filename), \
               'BSDS label not found at: {}'.format(filename)

        data = sio.loadmat(filename)
        gt = data['groundTruth']
        nGt = gt.shape[1]
        
        # return an empty list if no annotation is found
        if nGt == 0:
            return {'label'  : [], 
                    'width'  : 0,
                    'height' : 0}

        label = gt[0,0]['Boundaries'][0,0]

        for i in range(1, nGt):
            curGt = gt[0,i]
            curGt = curGt['Boundaries'][0,0]
            label += curGt

        label = label.astype(np.float32)
        label /= np.float32(nGt)

        # we assume the image has the same shape as its labels
        return {'label'  : label, 
                'width'  : label.shape[1],
                'height' : label.shape[0]}

    def process_label(self, label):
        ''' 
        Threshold the edge map for training
        '''
        # ignore weak boundaries during training
        label[np.logical_and(label>0.0, label<self.config['label_thrs'])] = -1.0

        # mark strong boundaries as 1
        label[label>=self.config['label_thrs']] = 1.0

        # we will throw any edge map that does not have edges
        if np.max(label) <= 0.0:
            label = []

        return label

    def process_output(self, score):
        # for edge detection, simply return the raw scores
        return score

    def write_output_at(self, label, i, output_dir):
        '''
        Write all edges into png files 
        '''
        # check output folder

        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        
        filename = os.path.join(output_dir, self._image_index[i] + '.png')

        # process label data
        label[label<0.0] = 0
        label[label>1.0] = 1.0
        label = (65535.0*label).astype(np.uint16)

        # now we can write png files (in 16bit png)
        cv2.imwrite(filename, label)

    def _apply_nms(self, input_dir, output_dir):
        '''
        Apply nms on raw edge output. NMS results are later used for eval
        '''
        # get all output edge files
        label_files = glob.glob(os.path.join(input_dir, '*.png'))

        # loop over each file and run nms (should be really fast)
        for label_file in label_files:
            # load file and convert into [0,1]
            edge_raw = cv2.imread(label_file, cv2.IMREAD_UNCHANGED)
            edge_raw = np.ascontiguousarray(edge_raw, dtype=np.float32)
            edge_raw /= 65535.0

            # nms over the result edge map
            orient = edgeOrient(edge_raw)
            orient = np.ascontiguousarray(orient, dtype=np.float32)
            edgeNms(edge_raw, orient, 1, 5, 1.01)
            edge_nms = (255.0*edge_raw).astype(np.uint8)

            # write to output folder
            filename = os.path.basename(label_file)
            cv2.imwrite(os.path.join(output_dir, filename), edge_nms)
        
    def _do_python_eval(self, output_dir = 'output'):
        '''
        We port the BSDS evaluation code into python
        '''
        
        # run nms over all images within the output folder 
        input_folder = output_dir
        nms_folder = os.path.join(output_dir, 'nms')

        if not os.path.isdir(nms_folder):
            os.mkdir(nms_folder)

        self._apply_nms(input_folder, nms_folder)

        # call python evaluation code (after NMS)
        # NOTE: by default, this will occupy all cores
        gt_folder = os.path.join(self._data_path, 'groundTruth', self._image_set)
        max_threads = multiprocessing.cpu_count()
        ods, ap = edgesEvalDir(nms_folder, gt_folder, 
                    self.config['eval_thrs'], numThreads=max_threads)

        # AP is not going to be accurate with less thrs
        print('ODS = {:.4f}, AP = {:.4f}'.format(ods, ap))
        print('~~~~~~~~')
        print('')
        print('--------------------------------------------------------------')
        print('Results computed with the **unofficial** Python eval code.')
        print('Results should be very close to the official MATLAB eval code.')
        print('ODS should be farily close. AP is less accurate with less thresholds')
        print('-- Thanks, Yin Li')
        print('--------------------------------------------------------------')


    def evaluate_labeling(self, output_dir):
        self._do_python_eval(output_dir)

    def competition_mode(self, on):
        pass
