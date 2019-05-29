# distutils: language = c++
# distutils: sources = ./cpp/correspondPixels.cc, ./cpp/edgeNms.cc

__author__ = 'yinli'

# This is a "mini" python port of edge evaluation code for BSDS

# import cython, numpy and opencv
import os
import cython
import numpy as np
import cv2
cimport numpy as np

# intialized Numpy. must do.
np.import_array()

# Declare the prototype of the C functions 
cdef extern from "edgeNms.h":
    void c_edgeNms(float *E, const float *O, const int h, const int w, const int r, int s, const float m)

cdef extern from "correspondPixels.h":
    void c_correspondPixels(double *E1, double *E2, const int h, const int w, const double maxDist, const double outlierCost, double* match1, double* match2)

def correspondPixels(np.ndarray[np.double_t, ndim=2, mode="c"] e1 not None, \
    np.ndarray[np.double_t, ndim=2, mode="c"] e2 not None, double maxDist):
    # define all the C-types
    cdef int h, w
    cdef double outlierCost
    cdef np.ndarray[np.double_t, ndim=2, mode='c'] m1
    cdef np.ndarray[np.double_t, ndim=2, mode='c'] m2

    # set up default parameters
    outlierCost = 100.0
    h, w = e1.shape[0], e1.shape[1]
    assert (h==e2.shape[0] and w==e2.shape[1])
    m1 = np.zeros_like(e1)
    m1 = np.ascontiguousarray(m1)
    m2 = np.zeros_like(e1)
    m2 = np.ascontiguousarray(m2)

    # call the underlying c function
    c_correspondPixels(&e1[0,0], &e2[0,0], h, w, maxDist, outlierCost, &m1[0,0], &m2[0,0])

    return (m1, m2)

def edgeNms(np.ndarray[float, ndim=2, mode="c"] edge not None, \
    np.ndarray[float, ndim=2, mode="c"] orient not None, int r, int s, float m):
    '''NMS for edge map'''
    cdef int h, w
    h, w = edge.shape[0], edge.shape[1]
    assert h == orient.shape[0] and w == orient.shape[1]
    c_edgeNms(&edge[0,0], &orient[0,0], h, w, r, s, m)
    return True

# generate convtri kernel
def convKernel(r):
    if r<=1:
        p = (12/r)/(r+2)-2
        kernel = [1, p, 1]
        kernel = np.asarray(kernel, dtype=np.float32)
        kernel = kernel / (2+p)
    else:
        kernel = range(1,r+1) + [r+1] + range(r, 0, -1)
        kernel = np.asarray(kernel, dtype=np.float32)
        kernel = kernel / ((r+1)*(r+1))
    return (kernel, r)

# trianglar convolution for smoothing
def convTri(img, r):
    kernel, r = convKernel(r)
    result = np.zeros_like(img)
    cv2.sepFilter2D(img, -1, kernel, kernel, result, (-1,-1), 0, cv2.BORDER_REFLECT)
    return result

# get the orientation of the edge map
def edgeOrient(edge):
    # smooth the edge a bit
    Oy, Ox = np.gradient(convTri(edge, 4))
    Oyx, Oxx = np.gradient(Ox)
    Oyy, Oxy = np.gradient(Oy)
    O = np.arctan(Oyy*np.sign(-Oxy)/(Oxx + 1e-5))
    O = O % np.pi
    return O
