import numpy as np
import os

video_dir = '/media/meerahahn/General_Datasets/EDGTEA/videos/'
for video in os.listdir(video_dir):
    vid = video[:-4]
    print vid
    cmmd = 'mkdir /media/meerahahn/General_Datasets/EDGTEA/hand_masks/' + vid
    os.system(cmmd)
    cmmd = 'python ./tools/demo_video.py --gpu 0 --def ./models/GTEA/hed/test.prototxt --net ./models/GTEA/hed/hand_vgg16_iter_36000.caffemodel --video ' + video_dir + video + ' --output /media/meerahahn/General_Datasets/EDGTEA/hand_masks/' + vid + ' --vis'
    os.system(cmmd)
