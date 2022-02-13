import numpy as np
import cv2 as cv
from pathlib import Path
from targets import Circles, Checkerboard
from classes import Line, ImageContainer

if __name__ == '__main__':
    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # create target pattern and image container
    target_pattern = Checkerboard(8, 11, 15)

    imgcon = ImageContainer("testimgs6/dngs", "*.dng")
    img_size = imgcon.imgsize

    # # Arrays to store object points and image points from all the images.
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