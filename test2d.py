# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 14:54:19 2021

@author: bekdulnm
"""

import numpy as np
import cv2 as cv
from pathlib import Path
from targets import Checkerboard
import matplotlib.pyplot as plt


# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points
objp = np.zeros((7*10,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:10].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = list(Path("twod images sep 27").glob('*.jpg'))

for fname in images:
    img = cv.imread(str(fname))
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (7,10), None)
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners)
    else:
        plt.imshow(img)
        break


ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

fx = mtx[0,0]
fy = mtx[1,1]

f = 4.401

mx = f/fx
my = f/fy

print(mx)
print(my)
