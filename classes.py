# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 16:31:52 2021

@author: bekdulnm
"""

import numpy as np
import cv2 as cv
from pathlib import Path
import transforms3d.reflections as tr
import transforms3d.affines as ta
from scipy.spatial.transform import Rotation as R
from targets import Checkerboard, Circles


class Line(object):
    """
    Line object in 3D.
    
    Parameters
    ----------
    origin : ndarray
        Any point belonging to the line.
    orientation : ndarray
        The direction of line.
    
    Attributes
    ----------
    origin : ndarray
        Any point belonging to the line.
    orientation : ndarray
        The normalized direction of line.
    """

    def __init__(self, origin, orientation):
        self.origin = np.array(origin)
        self.orientation = np.array(orientation)

    @property
    def orientation(self):
        return self._orientation

    @orientation.setter
    def orientation(self, orientation):
        self._orientation = orientation / np.linalg.norm(np.array(orientation))

    def intersect_ray(self, other):
        """
        Intersect this Line with another one.
        
        Implementation of the closest intersection point as described by the
        midpoint method [1]_.

        Parameters
        ----------
        other : Line
            The line to intersect with.

        Returns
        -------
        point : ndarray
            The midpoint where the two lines are nearest to each other.
        error : float
            The shortest distance between the two lines.
            
        .. [1] "Mid Point Method", wikipedia
            https://en.wikipedia.org/wiki/Skew_lines#Nearest_points.
        """
        # Rename variables according to wikipedia's convention.
        p1 = self.origin
        p2 = other.origin
        d1 = self.orientation
        d2 = other.orientation

        # Perform the midpont method.
        n = np.cross(d1, d2)
        n1 = np.cross(d1, n)
        n2 = np.cross(d2, n)
        c1 = p1 + np.dot((p2 - p1), n2) / np.dot(d1, n2) * d1
        c2 = p2 + np.dot((p1 - p2), n1) / np.dot(d2, n1) * d2

        # Find midpoint and error.
        point = (c1 + c2) / 2
        error = np.linalg.norm(c2 - c1)
        return point, error, c1, c2

    def intersect_point(self, point):
        line_to_point = np.array(point - self.origin)
        line_to_point = line_to_point / np.linalg.norm(line_to_point)
        if np.allclose(self.orientation, line_to_point) or \
                np.allclose(self.orientation, -line_to_point):
            ret = True
        else:
            ret = False
        return ret


class ImageContainer(object):
    """
    Image container objec with extracted points.
    
    Contains image locations, criteria for point detection and points for each
    image inside image directory.
    
    Parameters
    ----------
    path : str
        Path towards directory with images.
    
    Attributes
    ----------
    stereoimgs : Path
        Path object from pathlib library [1]_.
    criteria : tuple
        OpenCV defined criteria for corner points. The tuple comprises (1)
        openCV commands to define order, (2) max number of iterations, (3)
        min accuracy.
    objpoints_left : array_like
        List of object points on the left side of the images.
    objpoints_right : array like
        List of object points on the right side of the images.
    imgpoints_left : array_like
        List of image points on the left side of the images.
    imgpoints_right : array_like
        List of image points on the right side of the images.
    
    [1] https://docs.python.org/3/library/pathlib.html.
    """

    def __init__(self, path, ext):
        self.stereoimgs = list(map(str, list(Path(path).glob(ext))))
        img = cv.imread(self.stereoimgs[0])
        self.imgsize = (img.shape[0], img.shape[1])
        self.criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        self.objpoints = []
        self.imgpoints_left = []
        self.imgpoints_right = []

    @staticmethod
    def _img_split(img):
        """
        Split images on left and right sides.

        Parameters
        ----------
        img : ndarray, dtype=uint8
            Image in full color.

        Returns
        -------
        img_left : ndarray, dtype=uint8
            Image, where the right half is turned black.
        img_right : ndarray, dtype=uint8
            Image, where the left half is turned black.

        """
        mid_point = img.shape[1] // 2
        blck = np.zeros_like(img, dtype=np.uint8)

        img_left = blck.copy()
        img_right = blck.copy()

        img_left[:, :mid_point, :] = img[:, :mid_point, :]
        img_right[:, mid_point:, :] = img[:, mid_point:, :]
        return [img_left, img_right]

    def extract(self, target):
        """
        Extract points with target pattern.
        
        Extract points from images given the target pattern and append
        the extracted points and respective object points.

        Parameters
        ----------
        checkerboard : Checkerboard
            Checekerboard object with a single set of object points.

        Raises
        ------
        RuntimeError
            OpenCV cannot properly detect corners of the target.

        Returns
        -------
        None.

        """
        self.objpoints = target.gridpoints
        for img_path in self.stereoimgs:
            img = cv.imread(img_path)
            [img_left, img_right] = self._img_split(img)

            gray_img_left = cv.cvtColor(img_left, cv.COLOR_BGR2GRAY)
            gray_img_right = cv.cvtColor(img_right, cv.COLOR_BGR2GRAY)

            if isinstance(target, Checkerboard):
                ret_left, corners_left = \
                    cv.findChessboardCorners(gray_img_left,
                                             (target.verticals,
                                              target.horizontals),
                                             None)

                ret_right, corners_right = \
                    cv.findChessboardCorners(gray_img_right,
                                             (target.verticals,
                                              target.horizontals),
                                             None)

            elif isinstance(target, Circles):
                ret_left, corners_left = cv.findCirclesGrid(gray_img_left,
                                                            target.pattern_size,
                                                            flags=cv.CALIB_CB_ASYMMETRIC_GRID)

                ret_right, corners_right = cv.findCirclesGrid(gray_img_right,
                                                              target.pattern_size,
                                                              flags=cv.CALIB_CB_ASYMMETRIC_GRID)

            else:
                raise NotImplementedError("Target type is not implemented.")

            if ret_left is True and ret_right is True:
                self.imgpoints_left.append(corners_left)
                self.imgpoints_right.append(corners_right)
            else:
                raise RuntimeError("Target could not be detected properly." +
                                   f"\n img path = {img_path}")


class Homography(object):
    def __init__(self, rot_matrix, tran_vector, angles, mirror_points, intrinsic_matrix):
        self.rot_matrix = rot_matrix
        self.tran_vector = tran_vector
        self.rt_matrix = ta.compose(tran_vector, rot_matrix, [1, 1, 1])
        self.angles = angles
        self.dists = np.empty(4)
        self.mirror_normals = np.empty((4, 3))
        reflection_matrices = np.empty((4, 4, 4))

        for index, (angle, mirror_point) in enumerate(zip(angles, mirror_points)):
            self.mirror_normals[index] = np.array([np.cos(angle[1]) * np.sin(angle[0]),
                                                   np.sin(angle[1]) * np.sin(angle[0]),
                                                   np.cos(angle[0])])
            self.dists[index] = self._find_r(self.mirror_normals[index], mirror_point)
            reflection_matrices[index, :, :] = tr.rfnorm2aff(self.mirror_normals[index], mirror_point)

        self.left_refl_matrix = reflection_matrices[0, :, :].dot(reflection_matrices[1, :, :])
        self.right_refl_matrix = reflection_matrices[2, :, :].dot(reflection_matrices[3, :, :])
        self.intrinsic_matrix = intrinsic_matrix

        self.left_projection_matrix = self.intrinsic_matrix.dot(self.left_refl_matrix.dot(self.rt_matrix))
        self.right_projection_matrix = self.intrinsic_matrix.dot(self.right_refl_matrix.dot(self.rt_matrix))

    def project_to_image(self, world_points, rot_vecs, tran_vecs):
        left_points_list = []
        right_points_list = []
        for (rot_vec, tran_vec) in zip(rot_vecs, tran_vecs):
            r = R.from_rotvec(rot_vec)
            rt_matrix = ta.compose(np.array(tran_vec), r.as_matrix(), [1, 1, 1])
            left_projection_matrix, right_projection_matrix = self.construct_projection_matrix(rt_matrix)
            left_projected_points = np.empty((len(world_points), 2))
            right_projected_points = left_projected_points.copy()
            for i, point in enumerate(world_points):
                point = np.append(point, 1)
                left_projected_points[i] = self.non_homogeneous(left_projection_matrix.dot(point))
                right_projected_points[i] = self.non_homogeneous(right_projection_matrix.dot(point))
            left_points_list.append(left_projected_points)
            right_points_list.append(right_projected_points)
        return left_points_list, right_points_list

    def construct_projection_matrix(self, rt_matrix):
        left_projection_matrix = self.intrinsic_matrix.dot(self.left_refl_matrix.dot(rt_matrix))
        right_projection_matrix = self.intrinsic_matrix.dot(self.right_refl_matrix.dot(rt_matrix))
        return left_projection_matrix, right_projection_matrix

    # EXPERIMENTAL
    # def update(self, parameters):
    #     # SCALING
    #     tempvars = parameters[:8] * 1E4
    #
    #     angles = tempvars.reshape((4, 2))
    #     dists = parameters[8:12]
    #     [fx, fy, cx, cy,
    #      r1, r2, r3,
    #      tx, ty, tz] = parameters[12:]
    #
    #     r = R.from_rotvec([r1, r2, r3])
    #     self.rt_matrix = ta.compose(np.array([tx, ty, tz]), r.as_matrix(), [1, 1, 1])
    #
    #     mirror_normals = np.empty((4, 3))
    #     reflection_matrices = np.empty((4, 4, 4))
    #     for index, angle in enumerate(angles):
    #         mirror_normals[index] = np.array([np.cos(angle[1]) * np.sin(angle[0]),
    #                                           np.sin(angle[1]) * np.sin(angle[0]),
    #                                           np.cos(angle[0])])
    #         mirror_point = mirror_normals[index] * dists[index]
    #         reflection_matrices[index, :, :] = tr.rfnorm2aff(mirror_normals[index], mirror_point)
    #
    #     self.left_refl_matrix = reflection_matrices[0, :, :].dot(reflection_matrices[1, :, :])
    #     self.right_refl_matrix = reflection_matrices[2, :, :].dot(reflection_matrices[3, :, :])
    #
    #     self.intrinsic_matrix = np.array([[fx, 0, cx, 0],
    #                                       [0, fy, cy, 0],
    #                                       [0, 0, 1, 0]])
    #
    #     self.left_projection_matrix = self.intrinsic_matrix.dot(self.left_refl_matrix.dot(self.rt_matrix))
    #     self.right_projection_matrix = self.intrinsic_matrix.dot(self.right_refl_matrix.dot(self.rt_matrix))
    #
    # def get_parameters(self):
    #     r = R.from_matrix(self.rot_matrix)
    #     rot_vector = r.as_rotvec()
    #     return np.hstack([self.angles.flatten() / 1E4,
    #                       self.dists,
    #                       self.intrinsic_matrix[0, 0],
    #                       self.intrinsic_matrix[1, 1],
    #                       self.intrinsic_matrix[0, 2],
    #                       self.intrinsic_matrix[1, 2],
    #                       rot_vector,
    #                       self.tran_vector])

    # def update(self, parameters, scaling):
    #     # SCALING
    #     tempvars = parameters.copy() * scaling
    #
    #     angles = tempvars[:8].reshape((4, 2))
    #     dists = parameters[8:12]
    #     [fx, fy, cx, cy,
    #      r1, r2, r3,
    #      tx, ty, tz] = parameters[12:]
    #
    #     r = R.from_rotvec([r1, r2, r3])
    #     self.rt_matrix = ta.compose(np.array([tx, ty, tz]), r.as_matrix(), [1, 1, 1])
    #
    #     mirror_normals = np.empty((4, 3))
    #     reflection_matrices = np.empty((4, 4, 4))
    #     for index, angle in enumerate(angles):
    #         mirror_normals[index] = np.array([np.cos(angle[1]) * np.sin(angle[0]),
    #                                           np.sin(angle[1]) * np.sin(angle[0]),
    #                                           np.cos(angle[0])])
    #         mirror_point = mirror_normals[index] * dists[index]
    #         reflection_matrices[index, :, :] = tr.rfnorm2aff(mirror_normals[index], mirror_point)
    #
    #     self.left_refl_matrix = reflection_matrices[0, :, :].dot(reflection_matrices[1, :, :])
    #     self.right_refl_matrix = reflection_matrices[2, :, :].dot(reflection_matrices[3, :, :])
    #
    #     self.intrinsic_matrix = np.array([[fx, 0, cx, 0],
    #                                       [0, fy, cy, 0],
    #                                       [0, 0, 1, 0]])
    #
    #     self.left_projection_matrix = self.intrinsic_matrix.dot(self.left_refl_matrix.dot(self.rt_matrix))
    #     self.right_projection_matrix = self.intrinsic_matrix.dot(self.right_refl_matrix.dot(self.rt_matrix))

    # def get_parameters(self, scaling):
    #     r = R.from_matrix(self.rot_matrix)
    #     rot_vector = r.as_rotvec()
    #     return np.hstack([self.angles.flatten(),
    #                       self.dists,
    #                       self.intrinsic_matrix[0, 0],
    #                       self.intrinsic_matrix[1, 1],
    #                       self.intrinsic_matrix[0, 2],
    #                       self.intrinsic_matrix[1, 2],
    #                       rot_vector,
    #                       self.tran_vector]) / scaling

    def update(self, parameters):
        # theta_left_in, phi_left_in, r_left_in,
        #          theta_left_ou, phi_left_ou, r_left_ou,
        #          theta_right_in, phi_right_in, r_right_in,
        #          theta_right_ou, phi_right_ou, r_right_ou,
        # [theta_left_in, phi_left_in, theta_left_ou, phi_left_ou,
        #  theta_right_in, phi_right_in, theta_right_ou, phi_right_ou] =
        angles = parameters[:8].reshape((4, 2))
        dists = parameters[8:12]
        [fx, fy, cx, cy,
         r1, r2, r3,
         tx, ty, tz] = parameters[12:22]

        r = R.from_rotvec([r1, r2, r3])
        self.rt_matrix = ta.compose(np.array([tx, ty, tz]), r.as_matrix(), [1, 1, 1])

        mirror_normals = np.empty((4, 3))
        reflection_matrices = np.empty((4, 4, 4))
        for index, angle in enumerate(angles):
            mirror_normals[index] = np.array([np.cos(angle[1]) * np.sin(angle[0]),
                                              np.sin(angle[1]) * np.sin(angle[0]),
                                              np.cos(angle[0])])
            mirror_point = mirror_normals[index] * dists[index]
            reflection_matrices[index, :, :] = tr.rfnorm2aff(mirror_normals[index], mirror_point)

        self.left_refl_matrix = reflection_matrices[0, :, :].dot(reflection_matrices[1, :, :])
        self.right_refl_matrix = reflection_matrices[2, :, :].dot(reflection_matrices[3, :, :])

        self.intrinsic_matrix = np.array([[fx, 0, cx, 0],
                                          [0, fy, cy, 0],
                                          [0, 0, 1, 0]])

        self.left_projection_matrix = self.intrinsic_matrix.dot(self.left_refl_matrix.dot(self.rt_matrix))
        self.right_projection_matrix = self.intrinsic_matrix.dot(self.right_refl_matrix.dot(self.rt_matrix))

    def get_parameters(self):
        r = R.from_matrix(self.rot_matrix)
        rot_vector = r.as_rotvec()
        return np.hstack([self.angles.flatten(),
                          self.dists,
                          self.intrinsic_matrix[0, 0],
                          self.intrinsic_matrix[1, 1],
                          self.intrinsic_matrix[0, 2],
                          self.intrinsic_matrix[1, 2],
                          rot_vector,
                          self.tran_vector])

    @staticmethod
    def _find_r(normal, point):
        return np.array(normal).dot(point)
    
    @staticmethod
    def non_homogeneous(point):
        return np.array([point[0] / point[2], point[1] / point[2]])


class SingleCameraCalibrator(object):
    def __init__(self, path, ext, target):
        self.img_locations = list(map(str, list(Path(path).glob(ext))))
        img = cv.imread(self.img_locations[0])
        self.img_size = (img.shape[0], img.shape[1])
        self.criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        self.img_points = []
        self.obj_points = []

        for img_location in self.img_locations:
            img = cv.imread(img_location)
            img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            if isinstance(target, Checkerboard):
                ret, corners = cv.findChessboardCorners(img_gray,
                                                        (target.verticals, target.horizontals),
                                                        None)
            elif isinstance(target, Circles):
                ret, corners = cv.findCirclesGrid(img_gray,
                                                  target.pattern_size,
                                                  flags=cv.CALIB_CB_ASYMMETRIC_GRID)
            else:
                raise NotImplementedError("Target type is not implemented.")

            if ret is True:
                self.img_points.append(corners)
                self.obj_points.append(target.gridpoints)
            else:
                raise RuntimeError("Target could not be detected properly." + f"\n img path = {img_location}")

        rms_error, mtx, dist, rvecs, tvecs = cv.calibrateCamera(self.obj_points,
                                                                self.img_points,
                                                                img.shape[:-1],
                                                                None, None)
        self.rvecs = rvecs
        self.tvecs = tvecs
        h, w = self.img_size
        new_camera_mtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
        self.instrinsics = new_camera_mtx
        self.rms_error = rms_error

    def homogeneous_intrinsics(self):
        return np.hstack((self.instrinsics, np.zeros((3, 1))))


if __name__ == "__main__":
    cb = Checkerboard(8, 11, 15)
    imgcon = ImageContainer("testimgs")
    img_size = imgcon.imgsize
    imgcon.extract(cb)
