"""Set up paths for dense label."""
import os
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = os.path.dirname(__file__)

# Add caffe to PYTHONPATH
caffe_path = "/home/meerahahn/Documents/Software/caffe/python"
add_path(caffe_path)

# Add lib to PYTHONPATH
lib_path = os.path.join(this_dir, '..', 'lib')
add_path(lib_path)

# Control threading for BLAS
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '2'
