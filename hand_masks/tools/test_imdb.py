import _init_paths
from datasets.bsds import bsds
from datasets.gtea import gtea
from data_layer.roidb import prepare_roidb
import cv2
import numpy as np
import matplotlib.pyplot as plt

# imdb = bsds('train', '500')
# prepare_roidb(imdb)
# roidb = imdb.roidb
# print "Total number of images {:d}".format(len(roidb))
# for i in xrange(len(roidb)):
#     print "Image File Name: ", roidb[i]['image']

#     im = cv2.imread(roidb[i]['image'].encode('utf-8'))
#     label = roidb[i]['label'].astype(np.float32)
#     label[label<0] = 0

#     f, (ax1, ax2) = plt.subplots(1, 2)
#     ax1.imshow(im[:,:,::-1])
#     ax2.imshow(label, cmap=plt.get_cmap('Greys'))

#     plt.show()


imdb = gtea('train', '2016')
prepare_roidb(imdb)
roidb = imdb.roidb
print "Total number of images {:d}".format(len(roidb))
for i in xrange(len(roidb)):
    print "Image File Name: ", roidb[i]['image']

    im = cv2.imread(roidb[i]['image'].encode('utf-8'))
    label = roidb[i]['label'].astype(np.float32)
    label[label<0] = 0

    f, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(im[:,:,::-1])
    ax2.imshow(label, cmap=plt.get_cmap('gray'))

    plt.show()