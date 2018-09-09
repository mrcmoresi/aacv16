# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import division

import os
import cv2
import argparse
import pickle
import numpy as np

DETECTOR = cv2.ORB_create(nfeatures=2000, scaleFactor=1.2, edgeThreshold=1, fastThreshold=10)
DESCRIPTOR = DETECTOR

# DETECTOR = cv2.BRISK_create()
# DESCRIPTOR = DETECTOR

# DETECTOR = cv2.MSER_create(_min_area=10, _max_variation=0.25, _min_diversity=0.25)
# DESCRIPTOR = cv2.ORB_create()

# DETECTOR = cv2.xfeatures2d.SIFT_create()
# DESCRIPTOR = DETECTOR

# DETECTOR = cv2.xfeatures2d.SURF_create()
# DESCRIPTOR = DETECTOR


def kp2arr(kps):
    '''cv2.KeyPoint to np.array so that we can use pickle to save the model to disk'''
    return np.array([(kp.pt[0], kp.pt[1], kp.size, kp.angle) for kp in kps])


def arr2kp(arr):
    '''np.array to cv2.KeyPoint'''
    return [cv2.KeyPoint(x, y, size, angle) for (x, y, size, angle) in arr]


def train_model(path):
    '''
    Model training. path is a directoy with the images of the object of
    interest. The model is saved to path/model.dat
    '''

    model = {'keypoints': [], 'descriptors': [], 'bbox': []}
    for f in os.listdir(path):
        if os.path.splitext(f)[1] not in ('.jpeg', '.jpg', '.png'):
            continue
        filename = os.path.join(path, f)
        img = cv2.imread(filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        keypoints = DETECTOR.detect(img, None)
        keypoints, descriptors = DESCRIPTOR.compute(img, keypoints)

        # model = list of keypoints and descriptors
        model['keypoints'].append(kp2arr(keypoints))
        model['descriptors'].append(descriptors)

        # the spatial extent of the object is given by the bounding box of
        # keypoint locations
        x_min, y_min = np.min(model['keypoints'][-1][:, :2], axis=0)
        x_max, y_max = np.max(model['keypoints'][-1][:, :2], axis=0)
        model['bbox'].append((x_min, y_min, x_max, y_max))

        print('{} ({}x{}): {} features'.format(filename, img.shape[1], img.shape[0], len(keypoints)))

        visu = cv2.drawKeypoints(img, keypoints, None, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imshow("view", visu)
        cv2.waitKey()

    modelfile = os.path.join(path, 'model.dat')
    fnode = open(modelfile, 'w')
    pickle.dump(model, fnode)
    print('model saved to {}'.format(modelfile))



def eval_model(path, dist_threshold, min_matches, max_reprojection_error):
    '''Simple instance recognition (matching + homography estimation).
    path is the directory that containd the model.dat file
    dist_treshold is the matching distance threshold.
    min_matches is the minimum number of matching descriptors allowed.
    max_reprojection_error is the RANSAC threshold.
    '''
    modelfile = os.path.join(path, 'model.dat')
    model = pickle.load(open(modelfile))
    model['keypoints'] = [arr2kp(arr) for arr in model['keypoints']]

    # setup matcher (HAMMING distance for binary descriptors. NORM_L2 for real
    # ones). crossCheck is an alternative to the Lowe's distance ratio criteria
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    cam = cv2.VideoCapture(-1)
    while True:
        OK, img = cam.read()
        if not OK:
            break
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # detect + describe local features
        keypoints = DETECTOR.detect(img, None)
        keypoints, descriptors = DESCRIPTOR.compute(img, keypoints)

        # 1-NN descriptors matching (query to model). For binary descriptors, it
        # is a good idea to check also the second best (2-NN).
        matches = matcher.match(descriptors, model['descriptors'][0])

        # apply a distance threshold to the matching set. matches is a list of
        # cv2.DMatch objects, with fields: distance, queryIdx, trainIdx. The
        # indices are wrt the lists passed as arguments in matcher.match.
        good_matches = [m for m in matches if m.distance < dist_threshold]

        visu = np.dstack([img, img, img]) # just for visualization. gray->bgr

        # make sure a min number of keypoints has been matched against the model
        if len(good_matches) > min_matches:
            query_all = keypoints
            query_pts = np.float32([query_all[m.queryIdx].pt for m in good_matches])

            model_all = model['keypoints'][0]
            model_pts = np.float32([model_all[m.trainIdx].pt for m in good_matches])

            query_matched = [query_all[m.queryIdx] for m in good_matches]
            visu = cv2.drawKeypoints(visu, query_matched, None, (255, 0, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

            # estimate the model-to-query transformation
            H, mask = cv2.findHomography(model_pts, query_pts, method=cv2.RANSAC,
                                         ransacReprojThreshold=max_reprojection_error)
            H /= H[2][2]  # normalize matrix elements

            if np.sum(mask) > min_matches:
                mask = mask.ravel()
                inliers = [query_matched[i] for i in range(len(mask)) if mask[i] == 1]
                visu = cv2.drawKeypoints(visu, inliers, None, (0, 255, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

                # get the model bounting box and map it to the query image using H
                x1, y1, x2, y2 = model['bbox'][0]
                pA = H.dot(np.array([[x1], [y1], [1.0]])); pA /= pA[-1]
                pB = H.dot(np.array([[x2], [y1], [1.0]])); pB /= pB[-1]
                pC = H.dot(np.array([[x2], [y2], [1.0]])); pC /= pC[-1]
                pD = H.dot(np.array([[x1], [y2], [1.0]])); pD /= pD[-1]

                cv2.line(visu, (pA[0], pA[1]), (pB[0], pB[1]), (0, 0, 255), 2)
                cv2.line(visu, (pB[0], pB[1]), (pC[0], pC[1]), (0, 0, 255), 2)
                cv2.line(visu, (pC[0], pC[1]), (pD[0], pD[1]), (0, 0, 255), 2)
                cv2.line(visu, (pD[0], pD[1]), (pA[0], pA[1]), (0, 0, 255), 2)
            else:
                print('detection not reliable ({} inliers)'.format(np.sum(mask)))
        else:
            print('too few matches')

        cv2.imshow("view", visu)

        ch = cv2.waitKey(1)
        if ch in (ord('q'), ord('Q')):
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Simple keypoint-based instance recognition',
        add_help=True,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('task', help='train / eval', type=str)
    parser.add_argument('path', help='directory with the images of the object. If task=eval, this directory must contain a model.dat file', type=str)
    parser.add_argument('--dist_threshold', help='matching distance threshold', type=float, default=32)
    parser.add_argument('--min_matches', help='if #matches less than this value, detection not reliable', type=float, default=10)
    parser.add_argument('--max_reprojection_error', help='RANSAC reprojection error threshold', type=float, default=5.0)

    args = vars(parser.parse_args())

    if args['task'] == 'train':
        train_model(args['path'])
    elif args['task'] == 'eval':
        eval_model(args['path'], args['dist_threshold'], args['min_matches'], args['max_reprojection_error'])
    else:
        raise RuntimeError('wrong task')
