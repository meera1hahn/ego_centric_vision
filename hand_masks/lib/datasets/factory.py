# --------------------------------------------------------
# Dense Label Toolbox
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""

__sets = {}

#from datasets.pascal_voc import pascal_voc
#from datasets.camvid import camvid
#from datasets.kitti_seg import kitti_seg
#from datasets.cityscapes import cityscapes
from datasets.bsds import bsds
from datasets.gtea import gtea

# we currently support the following datasets:
# PASCAL VOC segmentation (object segmentation)
# Cityscapes (semantic segmentation)
# Camvid (semantic segmentation)
# Kitti (semantic segmentation)
# BSDS (edge detection)

# Set up voc_<year>_<split> 
#for year in ['2007', '2012']:
#    for split in ['train', 'val', 'trainval', 'test']:
#        name = 'voc_{}_{}'.format(year, split)
#        __sets[name] = (lambda split=split, year=year: pascal_voc(split, year))

# Set up cityscapes_2016_<split>
#for year in ['2016']:
#    for split in ['train', 'val', 'trainval', 'test']:
#        name = 'cityscapes_{}_{}'.format(year, split)
#        __sets[name] = (lambda split=split, year=year: cityscapes(split, year))

# Set up kitti_2015_<split>
#for year in ['2015']:
#    for split in ['train', 'val', 'trainval', 'test']:
#        name = 'kitti_seg_{}_{}'.format(year, split)
#        __sets[name] = (lambda split=split, year=year: kitti_seg(split, year))

# Set up camvid_2007_<split>
#for year in ['2007']:
#    for split in ['train', 'val', 'trainval', 'test']:
#        name = 'camvid_{}_{}'.format(year, split)
#        __sets[name] = (lambda split=split, year=year: camvid(split, year))

# Set up bsds_500_<split>
for year in ['500']:
    for split in ['train', 'val', 'test']:
        name = 'bsds_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: bsds(split, year))

for year in ['2016']:
    for split in ['train', 'test']:
        name = 'gtea_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: gtea(split, year))


def get_imdb(name):
    """Get an imdb (image database) by name."""
    if not __sets.has_key(name):
        raise KeyError('Unknown dataset: {}'.format(name))
    return __sets[name]()

def list_imdbs():
    """List all registered imdbs."""
    return __sets.keys()
