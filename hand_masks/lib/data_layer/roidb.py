# --------------------------------------------------------
# Dense Label Toolbox
# --------------------------------------------------------

"""Transform a roidb into a trainable roidb by adding a bunch of metadata."""

import numpy as np
from dense_label.config import cfg

def prepare_roidb(imdb):
    """Enrich the imdb's roidb by adding some derived quantities that
    are useful for training. 
    """
    valid_im_info = imdb.roidb[0].has_key('width') and imdb.roidb[0].has_key('height')
    if not valid_im_info:
        sizes = [PIL.Image.open(imdb.image_path_at(i)).size
                 for i in xrange(imdb.num_images)]
    else: 
        print "Skipping im_info for roidb"
        
    roidb = imdb.roidb
    for i in xrange(len(imdb.image_index)):

        if not valid_im_info:
            roidb[i]['width'] = sizes[i][0]
            roidb[i]['height'] = sizes[i][1]

        # call the imdb process label function
        roidb[i]['label'] = imdb.process_label(roidb[i]['label'])
        # link image path in roidb
        roidb[i]['image'] = imdb.image_path_at(i)


