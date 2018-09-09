
from __future__ import print_function
from __future__ import division

import sys

from os import listdir, makedirs
from os.path import join, splitext, abspath, split, exists

import numpy as np
np.seterr(all='raise')

import cv2

from skimage.feature import daisy
from skimage.transform import rescale

from sklearn.svm import LinearSVC

from scipy.spatial import distance
from scipy.io import savemat, loadmat


SCALES_3 = [1.0]
SCALES_5 = [1.0, 0.707, 0.5, 0.354, 0.25]
def extract_multiscale_dense_features(imfile, step=8, scales=SCALES_3):
    im = cv2.imread(imfile, cv2.IMREAD_GRAYSCALE)
    feat_all = []
    for sc in scales:
        dsize = (int(sc * im.shape[1]), int(sc * im.shape[0]))
        im_scaled = cv2.resize(im, dsize, interpolation=cv2.INTER_LINEAR)
        feat, img = daisy(im_scaled, step=step, normalization='l2', visualize=True)
        cv2.imshow('Imagen', img)
  
        if feat.size == 0:
            break
        ndim = feat.shape[2]
        feat = np.atleast_2d(feat.reshape(-1, ndim))
        feat_all.append(feat)
    return np.row_stack(feat_all).astype(np.float32)


if __name__ == "__main__":
    test = extract_multiscale_dense_features("mono-001.jpg")