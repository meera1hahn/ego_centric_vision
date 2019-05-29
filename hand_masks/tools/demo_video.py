import _init_paths
from dense_label.test import im_label
from dense_label.config import cfg, cfg_from_file, cfg_from_list
import datasets.imdb
import argparse
import pprint
import time, os, sys
import glob
import cv2
import numpy as np
from skimage import measure
caffe_root = '/home/meerahahn/Documents/Software/caffe/python'
sys.path.append(caffe_root)
import caffe

def post_process_hands(scores, vfunc=None, 
						thresh=None, area_thresh=None):
	'''
	Post-Processing hand scores
	'''
	h, w = scores.shape[0], scores.shape[1]
	if vfunc is None:
		vfunc = np.ones((h, w), dtype=np.float)

	if thresh is None:
		thresh = 0.8

	if area_thresh is None:
		area_thresh = 0.01 * scores.size

	# copy labels
	org_scores = scores.copy()

	# binarize the scores
	binary_label = scores>thresh
	label = measure.label(binary_label)
	masked_label = label[(vfunc>0.8)]

	# loop over masked cc and compute 
	num_labels = np.max(label)
	masked_cc_areas = np.zeros((num_labels,))
	for i in xrange(1, num_labels+1):
		masked_cc_areas[i-1] = np.sum(masked_label==i)
	
	# sort the area size
	area_idx = np.argsort(-masked_cc_areas)
	final_label = np.zeros_like(label)

	# pick top 2 segment
	# also throw away small regions
	for idx in area_idx[:2]:
		if masked_cc_areas[idx] > area_thresh:
			final_label[label==(idx+1)] = 1

	org_scores[final_label==0] = 0

	return org_scores

def process_video_folder(net, video_folder, vis, vfunc, output_folder):

	# open videos and run segmentation
	video_frames = sorted(glob.glob(os.path.join(video_folder, '*.jpg')))

	# open and label every frame
	for video_frame in video_frames:
		im = cv2.imread(video_frame)

		if cfg.TEST.USE_FLIPPED:
			im_flip = im[:, ::-1, :]

		# detection on the orginal image
		scores = im_label(net, im)

		# flip the image if data augmentation is turned on during testing
		if cfg.TEST.USE_FLIPPED:
			scores_flip = im_label(net, im_flip)
			scores = 0.5*(scores + scores_flip[:, ::-1])

		output_file = os.path.join(output_folder, 
					os.path.basename(video_frame).split('.')[0] + '.png')

		print "Processing {:s}".format(output_file)

		# post processing hand scores
		label = post_process_hands(scores, vfunc)

		# rescale label and save
		label = (255.0*label).astype(np.uint8)

		if vis:
			vis_im = im.copy()
			vis_im[:,:,0] = 0.5*vis_im[:,:,0] + 0.5* label
			cv2.imshow('image',vis_im)
			cv2.waitKey(10)
		else:
			cv2.imwrite(output_file, label)

	if vis:
		cv2.destroyAllWindows()

def process_video_file(net, video_file, vis, vfunc, output_folder):

	cap = cv2.VideoCapture(video_file)
	if not cap.isOpened():
		print "Can not open video file: {:s}".format(video_file)

	frame_num = 0

	while True:
		# fetch frame from video
		ret, im = cap.read()
		if (not ret) or (im is None):
			break

		if cfg.TEST.USE_FLIPPED:
			im_flip = im[:, ::-1, :]

		# detection on the orginal image
		scores = im_label(net, im)

		# flip the image if data augmentation is turned on during testing
		if cfg.TEST.USE_FLIPPED:
			scores_flip = im_label(net, im_flip)
			scores = 0.5*(scores + scores_flip[:, ::-1])

		output_file = os.path.join(output_folder, 
					'{:06d}.png'.format(frame_num))

		print "Processing {:s}".format(output_file)

		# post processing hand scores
		label = post_process_hands(scores, vfunc)

		# rescale label and save
		label = (255.0*label).astype(np.uint8)

		#if vis:
		#	vis_im = im.copy()
		#	vis_im[:,:,0] = 0.5*vis_im[:,:,0] + 0.5* label
		#	cv2.imshow('image',vis_im)
		#	cv2.waitKey(10)
		#else:
		cv2.imwrite(output_file, label)

		frame_num += 1

def parse_args():
	"""
	Parse input arguments
	"""
	parser = argparse.ArgumentParser(description='Test a Fast R-CNN network')
	parser.add_argument('--gpu', dest='gpu_id', help='GPU id to use',
						default=0, type=int)
	parser.add_argument('--def', dest='prototxt',
						help='prototxt file defining the network',
						default=None, type=str)
	parser.add_argument('--net', dest='caffemodel',
						help='model to test',
						default=None, type=str)
	parser.add_argument('--cfg', dest='cfg_file',
						help='optional config file', default=None, type=str)
	parser.add_argument('--wait', dest='wait',
						help='wait until net file exists',
						default=True, type=bool)
	parser.add_argument('--video', dest='video_file',
						help='folder to video frames',
						default='video_frames', type=str)
	parser.add_argument('--output', dest='output_folder', help='output folder',
						default='video_output', type=str)
	parser.add_argument('--vis', dest='vis', help='visualize labelling',
						action='store_true')
	parser.add_argument('--vfunc', dest='vfunc',
						help='vignetting mask',
						default=None, type=str)

	if len(sys.argv) == 1:
		parser.print_help()
		sys.exit(1)

	args = parser.parse_args()
	return args

if __name__ == '__main__':
	args = parse_args()

	# set up caffe net
	print('Called with args:')
	print(args)

	if args.cfg_file is not None:
		cfg_from_file(args.cfg_file)

	cfg.GPU_ID = args.gpu_id
	video_file = args.video_file
	output_folder = args.output_folder
	if not os.path.isdir(output_folder):
		os.mkdir(output_folder)

	print('Using config:')
	pprint.pprint(cfg)

	while not os.path.exists(args.caffemodel) and args.wait:
		print('Waiting for {} to exist...'.format(args.caffemodel))
		time.sleep(10)

	caffe.set_mode_gpu()
	caffe.set_device(args.gpu_id)
	net = caffe.Net(args.prototxt, args.caffemodel, caffe.TEST)

	# load vfunc for resoration
	if args.vfunc is not None:
		vfunc_img = cv2.imread(args.vfunc)
		vfunc = vfunc_img[:,:,0].astype(np.float)/255.0
	else:
		vfunc = args.vfunc
	# vis or store
	vis = args.vis

	if os.path.isdir(video_file):
		process_video_folder(net, video_file, vis, vfunc, output_folder)
	else:
		process_video_file(net, video_file, vis, vfunc, output_folder)
