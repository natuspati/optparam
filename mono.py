import numpy as np
import cv2 as cv
from pathlib import Path
from targets import Circles, Checkerboard
from classes import Line, ImageContainer
import rawpy
import imageio
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # create target pattern and image container
    target_pattern = Checkerboard(8, 11, 15)
    objp = target_pattern.gridpoints

    path = "testimgs7/dngs"
    ext = "*.DNG"

    lst = list(map(str, list(Path(path).glob(ext))))
    objpoints = []
    imgpoints = []

    for raw_img in lst:
        with rawpy.imread(raw_img) as raw:
            rgb = raw.postprocess()
            gray = cv.cvtColor(rgb, cv.COLOR_BGR2GRAY)
            # Find the chess board corners
            ret, corners = cv.findChessboardCorners(gray, (7, 10), None)
            # If found, add object points, image points (after refining them)
            if ret:
                objpoints.append(objp)
                corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
                imgpoints.append(corners)

    # for raw_img in lst:
    #     img = cv.imread(raw_img)
    #     gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    #     # Find the chess board corners
    #     ret, corners = cv.findChessboardCorners(gray, (7, 10), None)
    #     # If found, add object points, image points (after refining them)
    #     if ret:
    #         objpoints.append(objp)
    #         corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
    #         imgpoints.append(corners)

    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    with rawpy.imread(lst[0]) as raw:
        rgb = raw.postprocess()
        h, w = rgb.shape[:2]
        newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

        # undistort
        dst = cv.undistort(rgb, mtx, dist, None, newcameramtx)
        # # crop the image
        # x, y, w, h = roi
        # dst = dst[y:y + h, x:x + w]
        cv.imwrite('phoneraw.png', dst)

    # rgb = cv.imread(lst[0])
    # h, w = rgb.shape[:2]
    # newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    #
    # # undistort
    # dst = cv.undistort(rgb, mtx, dist, None, newcameramtx)
    # cv.imwrite('calibresultjpg.png', dst)

    # imgcon = ImageContainer("testimgs6/dngs", "*.DNG")
    # img_size = imgcon.imgsize

    # # Arrays to store object points and ]image points from all the images.
    # objpoints = [] # 3d point in real world space
    # imgpoints = [] # 2d points in image plane.
    # images = glob.glob('*.jpg')
    # for fname in images:
    #     img = cv.imread(fname)
    #     gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    #     # Find the chess board corners
    #     ret, corners = cv.findChessboardCorners(gray, (7,6), None)
    #     # If found, add object points, image points (after refining them)
    #     if ret == True:
    #         objpoints.append(objp)
    #         corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
    #         imgpoints.append(corners)
    #         # Draw and display the corners
    #         cv.drawChessboardCorners(img, (7,6), corners2, ret)
    #         cv.imshow('img', img)
    #         cv.waitKey(500)
    # cv.destroyAllWindows()