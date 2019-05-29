# Dense Label Toolbox #

### What is this repository for? ###

This is a toolbox for dense image labeling.
 
* Frontend: Python; Backend: Caffe
* Flexible IO and data augmentation
* Supports multiple tasks / datasets

### How do I get set up? ###
* Dependencies: pycaffe, easy_dict
* cd ./lib 
* make
* edit the caffe_path in ./tools/_init_paths_.py

### Example: Training Hand Segmentation on GTEA Gaze Plus ###
* Data preparation: link Hand2K dataset into ./data./GTEA_GAZE_PLUS
    * Folder organization: Train / Test -> Images / Masks -> Video Name -> Frames

* Download VGG16 folder and put it under ./data/imagenet_models/vgg16.caffemodel

* Training for 32000 iterations (~80 epochs) on GPU 0 using trainval set
```
#!shell
python ./tools/train_net.py --gpu 0 --solver models/GTEA/hed/solver.prototxt --weights data/imagenet_models/vgg16.caffemodel --iters 32000 --imdb gtea_2016_train

```

* Testing the trained model on GPU 0 using test set
```
#!shell
python ./tools/test_net.py --gpu 0 --def ./models/BSDS/hed/test.prototxt --net ./output/gtea_2016_train/hand_vgg16_iter_32000.caffemodel --imdb gtea_2016_test

```

* Testing the trained model on an input video (using GPU 0)
```
#!shell
python ./tools/demo_video.py --gpu 0 --def ./models/GTEA/hed/test.prototxt --net ./output/gtea_2016_train/hand_vgg16_iter_32000.caffemodel --video /media/yin/WORK/organized_datasets/GTEA_Gaze_Plus/videos/Rahul_Turkey.avi --vis

```

* A pre-trained hand model (on GTEA Gaze+) can be downloaded from [this link](https://dl.dropboxusercontent.com/u/39491694/hand_vgg16_iter_36000.caffemodel)


### Example: Training Edge Detection on BSDS ###
* Data preparation: link BSDS folder into ./data/bsds/. 
    * Images should be in ./data/bsds/data/images
    * Annotations (mat files) should be in ./data/bsds/data/groundTruth
    * Image and annotation folders should have sub-folders train/val/test


* Download VGG16 folder and put it under ./data/imagenet_models/vgg16.caffemodel


* Training for 2400 iterations (80 epochs) on GPU 0 using trainval set
```
#!shell
python ./tools/train_net.py --gpu 0 --solver models/BSDS/hed/solver.prototxt --weights data/imagenet_models/vgg16.caffemodel --iters 2400 --imdb bsds_500_train+bsds_500_val

```

* Testing the trained model on GPU 0 using test set
```
#!shell
python ./tools/test_net.py --gpu 0 --def ./models/BSDS/hed/test.prototxt --net ./output/bsds_500_train+bsds_500_val/hed_vgg16_iter_2400.caffemodel --imdb bsds_500_test --bench
