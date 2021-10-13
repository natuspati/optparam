#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 11:10:29 2021

@author: bekdulnm
"""

import cv2 as cv
import numpy as np
import glob
import csv
import matplotlib.pyplot as plt
from pathlib import Path


def img_split(img):
    mid_point = img.shape[1] // 2
    blck = np.zeros_like(img, dtype=np.uint8)
    
    img_left = blck.copy()
    img_right = blck.copy()
    
    img_left[:,:mid_point,:] = img[:,:mid_point,:]
    img_right[:,mid_point:,:] = img[:,mid_point:,:]
    return [img_left, img_right]


if __name__ == "__main__":
    
    # prepare ideal target points
    cb_v = 8 #checkerboard vertical number of checkers
    cb_h = 11 #checkerboard horizontal number of checkers
    cb_size = 15 #checkerboard size
    v_edge = cb_v - 1
    h_edge = cb_h - 1
    idealpoints = np.zeros((v_edge*h_edge,3), np.float32)
    idealpoints[:,:2] = cb_size*np.mgrid[0:v_edge,0:h_edge].T.reshape(-1,2)
    
    objpoints = [] # 3d point in real world space
    mono_imgpoints = [] # 2d points in image plane.

    # Termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # load set of 2d planar images for initial guess of the matrix
    mono_imgs = Path("twod images sep 27/").glob('*.JPG')

    for index, img_name in enumerate(mono_imgs):
        img = cv.imread(str(img_name))
        gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        ret, corners = cv.findChessboardCorners(gray_img, (v_edge,h_edge), None)
        if ret is True:
            objpoints.append(idealpoints)
            corners2 = cv.cornerSubPix(gray_img,
                                        corners,
                                        (11,11), (-1,-1),
                                        criteria)
            mono_imgpoints.append(corners)

            # cv.drawChessboardCorners(img, (v_edge,h_edge), corners2, ret)
            # cv.imshow('img', img)
            # cv.waitKey(1000)
        else:
            raise RuntimeError("Checkerboard could not be detected properly" +
                                f", img = {index}")

    # cv.destroyAllWindows()

    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints,
                                                      mono_imgpoints,
                                                      gray_img.shape[::-1],
                                                      None, None)

    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv.norm(mono_imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
        mean_error += error
    print( "total error: {}".format(mean_error/len(objpoints)) )

    # load stereo calibration images
    stereo_imgs = Path("testimgs").glob('*.jpg')
    objpoints1 = []
    objpoints2 = []
    left_imgs = []
    right_imgs = []
    imgpoints1 = []
    imgpoints2 = []
    
    for index, img_name in enumerate(stereo_imgs):
        img = cv.imread(str(img_name))
        [img_left, img_right] = img_split(img)

        gray_img_left = cv.cvtColor(img_left, cv.COLOR_BGR2GRAY)
        ret, corners = cv.findChessboardCorners(gray_img_left, (v_edge,h_edge), None)
        if ret is True:
            objpoints1.append(idealpoints)
            corners2 = cv.cornerSubPix(gray_img_left,
                                        corners,
                                        (11,11), (-1,-1),
                                        criteria)
            imgpoints1.append(corners)
            
            # cv.drawChessboardCorners(img, (v_edge,h_edge), corners2, ret)
            # cv.imshow('img', img)
            # cv.waitKey(1000)
        else:
            raise RuntimeError("Checkerboard could not be detected properly" +
                                f", img = {index}")

        gray_img_right = cv.cvtColor(img_right, cv.COLOR_BGR2GRAY)
        if ret is True:
            objpoints2.append(idealpoints)
            corners2 = cv.cornerSubPix(gray_img_right,
                                        corners,
                                        (11,11), (-1,-1),
                                        criteria)
            imgpoints2.append(corners)
        else:
            raise RuntimeError("Checkerboard could not be detected properly" +
                                f", img = {index}")

        left_imgs.append(img_left)
        right_imgs.append(img_right)

    imgsize = (gray_img.shape[0], int(gray_img.shape[1]/2))

    [retval, mtx1, dist1, mtx2, dist2,
      R12, T12, E12, F12] = cv.stereoCalibrate(objpoints1,
                                              imgpoints1,
                                              imgpoints2,
                                              mtx,
                                              dist,
                                              mtx,
                                              dist,
                                              imgsize)

    # cv.destroyAllWindows()

    # rectify coordinates to 1st camera CS
    [R1, R2, P1, P2, Q, validPixROI1, validPixROI2] = cv.stereoRectify(mtx1,
                                                                        dist1,
                                                                        mtx2,
                                                                        dist2,
                                                                        imgsize,
                                                                        R12,
                                                                        T12)

    # traingulate points from left and right images
    points4d = cv.triangulatePoints(P1, P2, imgpoints1[1][:,0,:].T, imgpoints2[1][:,0,:].T)

    # img = cv.imread(str(stereo_imgs[0]))
    # img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # halves = img_split(img)

    # for index, img_name in enumerate(halves):
    #     img = cv.imread(str(img_name))
    #     gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    #     ret, corners = cv.findChessboardCorners(gray_img, (v_edge,h_edge), None)
    #     if ret is True:
    #         objpoints.append(idealpoints)
    #         corners2 = cv.cornerSubPix(gray_img,
    #                                     corners,
    #                                     (11,11), (-1,-1),
    #                                     criteria)
    #         mono_imgpoints.append(corners)

    #         cv.drawChessboardCorners(img, (v_edge,h_edge), corners2, ret)
    #         cv.imshow('img', img)
    #         cv.waitKey(1000)
    #     else:
    #         raise RuntimeError("Checkerboard could not be detected properly" +
    #                             f", img = {index}")

    # cv.destroyAllWindows()

    # S = cv.stereoCalibrate(objpoints, imagePoints1, imagePoints2, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, imageSize)

    # imgpoints = np.reshape(packed_imgpoints,(2*np.shape(objpoints)[1],2))
    # pimgpoints1 = packed_imgpoints[0]
    # pimgpoints2 = packed_imgpoints[1]
    
    # # load test image and split it into two
    # path = Path("testimgs")
    # all_imgs = list(path.glob('*.jpg'))
    # img_name = str(all_imgs[0])
    # img = cv.imread(img_name)
    # img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # halves = img_split(img)
    

    

    # # calibration assuming both pathes come from single camera
    # ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, packed_imgpoints, gray.shape[::-1], None, None)

    # plt.close()
    # plt.figure()
    # plt.imshow(img_gray, cmap='gray', vmin=0, vmax=255)
    # plt.scatter(imgpoints[:,0], imgpoints[:,1], s=20, facecolors='none', edgecolors='r')
    # plt.show

