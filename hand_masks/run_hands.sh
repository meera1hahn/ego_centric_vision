VIDEO_ROOT='/media/meerahahn/EDGTEA/videos/'
python ./tools/demo_video.py --gpu 0 --def ./models/GTEA/hed/test.prototxt --net ./models/GTEA/hed/hand_vgg16_iter_36000.caffemodel --video $VIDEO_ROOT/test.mp4 --output /media/meerahahn/EDGTEA/handMaskOutput/ --vis
