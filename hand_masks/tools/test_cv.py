import os
import cv2


print os.path.isfile('/nethome/nruiz9/research/recipes/actions/dense-label/data/Ahmad_Snack.avi')
video_file = '/nethome/nruiz9/research/recipes/actions/dense-label/data/Ahmad_Snack.avi'
cap = cv2.VideoCapture(video_file)
print cap
if not cap.isOpened():
	print "Can not open video file: {:s}".format(video_file)
else:
        print cap.isOpened()
