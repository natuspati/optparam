# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 11:19:33 2021

@author: bekdulnm
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from targets import Circles, Checkerboard
from classes import Line, ImageContainer
import glob
import rawpy
import imageio


if __name__ == "__main__":
    import numpy as np


    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((3 * 3, 3), np.float32)
    objp[:, :2] = np.mgrid[0:3, 0:3].T.reshape(-1, 2)
    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.
    images = glob.glob('testimgs11/raw2d/*.JPG')
    for fname in images:
        img = cv.imread(fname)
    # fname = images[0]
    # with rawpy.imread(fname) as raw:
    #     img = raw.postprocess()
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, (3, 3), None)
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners)
#
    # ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    # img = cv.imread(images[0])
    # h, w = img.shape[:2]
    # newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    # # undistort
    # dst = cv.undistort(img, mtx, dist, None, newcameramtx)
    # # # crop the image
    # # x, y, w, h = roi
    # # dst = dst[y:y + h, x:x + w]
    #
    # fig, ax = plt.subplots()
    # ax.imshow(img)
    #
    # fig1, ax1 = plt.subplots()
    # ax1.imshow(dst)
