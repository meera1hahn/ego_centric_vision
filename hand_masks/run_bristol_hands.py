import numpy as np
import os

video_dir = '/home/meerahahn/Documents/General_Datasets/Bristol_Egocentric_Object_Interactions_Dataset_2014/Videos/'
for video in os.listdir(video_dir):
    vid = video[:-4]
    print vid
    cmmd = 'mkdir video_output/' + vid
    os.system(cmmd)
    cmmd = 'python ./tools/demo_video.py --gpu 0 --def ./models/GTEA/hed/test.prototxt --net ./models/GTEA/hed/hand_vgg16_iter_36000.caffemodel --video ' + video_dir + video + ' --output video_output/' + vid + ' --vis'
    os.system(cmmd)
    break
