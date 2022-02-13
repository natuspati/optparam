# -*- coding: utf-8 -*-
"""
Created on Sun Oct 24 15:45:02 2021

@author: bekdulnm
"""

# imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from scipy.spatial.transform import Rotation as R
from scipy.optimize import least_squares
from scipy.sparse import csc_matrix
from classes import Line, ImageContainer
from system import System, TargetContainer, Camera, Mirror, TargetSystem
from targets import Circles, Checkerboard
from plotfunctions import PlotContainer

# to be removed after finishing the optimization algorithm
# import scipy.optimize._lsq.trf as trf

# trf.passlist = []


def split_fun(original_str, superscript=1):
    split_str = original_str.split(',')
    imp_str = split_str[0]
    if 1 < len(imp_str) < 4 and len(split_str) != 1:
        new_str = f"{imp_str}^{superscript}" + f", {split_str[1]}"
    elif len(imp_str) > 3 and len(split_str) != 1:
        new_str = f"{imp_str[:-2]}^{superscript}" + f", {split_str[1]}"
    elif len(imp_str) < 4 and len(split_str) == 1:
        new_str = f"{original_str}^{superscript}"
    elif len(imp_str) > 3 and len(split_str) == 1:
        new_str = f"{imp_str[:-2]}^{superscript}"
    elif superscript > 9 and len(split_str) != 1:
        new_str = f"{imp_str[:-2]}" + "{" + f"{superscript}" + "}" + f", {split_str[1]}"
    elif superscript > 9 and len(split_str) == 1:
        new_str = f"{imp_str[:-2]}" + "{" + f"{superscript}" + "}"
    else:
        raise RuntimeError("String formatting is incorrect.")
    return new_str


def objfun(parameters, syst, targets, imgcon):
    objpoints = imgcon.objpoints
    no_imgs = len(targets.tlst)
    no_points = len(imgcon.objpoints)

    system_parameters = parameters[:17]
    syst.update(system_parameters)
    target_parameters = np.reshape(parameters[17:], (no_imgs, 6))  # check
    targets.update(target_parameters)

    reconstructed_impgpoints_left = []
    reconstructed_impgpoints_right = []

    for trgt in targets.tlst:
        reconstructed_left, reconstructed_right = reconstruct_image(syst,
                                                                    trgt,
                                                                    objpoints)
        reconstructed_impgpoints_left.append(reconstructed_left)
        reconstructed_impgpoints_right.append(reconstructed_right)

    error_vector = find_error_vector(reconstructed_impgpoints_left,
                                     reconstructed_impgpoints_right,
                                     imgcon,
                                     no_imgs,
                                     no_points)
    return error_vector


def reconstruct_image(system, target, objpoints):
    reconstructed_left = np.zeros((objpoints.shape[0], 2))
    reconstructed_right = np.zeros((objpoints.shape[0], 2))
    for i, objpoint in enumerate(objpoints):
        reconstructed_left[i], reconstructed_right[i] = reconstruct_point(system, target, objpoint)
    return reconstructed_left, reconstructed_right


def reconstruct_point2(system, target, objpoint):
    """
    Find reconstruction (u,v) of an object point (x, y, z).

    Find projection on the image (u, v) for a given object point (x, y, z). Finds reflected focal points for the left
    and right optical paths. Casts rays from object points to the reflect focal points. The rays are reflected by
    the mirrors and will go through the real focal point automatiaclly. Reflected rays are intersected with the sensor
    to obtain image projections (u, v).

    .. warning :: The other reconstruction function, :py:func:`reconstruct_point`, performs the same reconstructions
        but in a computationally more efficient manner.

    Parameters
    ----------
    system : System
        Camera - Mirror system.
    target : TargetSystem
        Taget system with target to global coordinate transformation.
    objpoint : ndarray
        Object point is the exact target geometry in local coordinate system.

    Returns
    -------
    projection_left, projection_right : ndarray
        Projections from left and right optical paths in image coordinate system (u, v).

    """
    # find left optical path
    line_left = target.create_ray_from_local(objpoint, system.focal_left)
    line_left1 = left_outer.reflect_ray(line_left)
    line_left2 = left_inner.reflect_ray(line_left1)

    # find right optical path
    line_right = target.create_ray_from_local(objpoint, system.focal_right)
    line_right1 = right_outer.reflect_ray(line_right)
    line_right2 = right_inner.reflect_ray(line_right1)

    # find intersection between left/right optical paths with sensor
    projection_left = system.cam.intersect_ray(line_left2)
    projection_right = system.cam.intersect_ray(line_right2)
    return projection_left, projection_right
    # return np.hstack((projection_left, projection_right))


def reconstruct_point(system, target, objpoint):
    """
    Find reconstruction (u,v) of an object point (x, y, z).

    Find projection on the image (u, v) for a given object point (x, y, z). Finds reflection of an object point in
    global coordinate system through the mirrors. Casts rays to the focal. Intersects with the sensor to obtain image
    projections (u, v).

    Parameters
    ----------
    system : System
        Camera - Mirror system.
    target : TargetSystem
        Taget system with target to global coordinate transformation.
    objpoint : ndarray
        Object point is the exact target geometry in local coordinate system.

    Returns
    -------
    projection_left, projection_right : ndarray
        Projections from left and right optical paths in image coordinate system (u, v).

    """
    # Map target to global coordinates.
    objpoint_global = target.to_global(objpoint)

    # find left optical path.
    point_left_outer = system.left_outer.reflect_point(objpoint_global)
    point_left_inner = system.left_inner.reflect_point(point_left_outer)
    projection_left = system.cam.project_point_on_sensor(point_left_inner)

    # find right optical path.
    point_right_outer = system.right_outer.reflect_point(objpoint_global)
    point_right_inner = system.right_inner.reflect_point(point_right_outer)
    projection_right = system.cam.project_point_on_sensor(point_right_inner)
    return projection_left, projection_right

def find_error_vector(reconstructed_right_side,
                      reconstructed_left_side,
                      imgcon, I, N):
    error_vector = np.zeros(2 * I * N)
    counter = 0

    for i in range(I):
        imgpoints_left = imgcon.imgpoints_left[i].reshape((N, 2))
        imgpoints_right = imgcon.imgpoints_right[i].reshape((N, 2))

        reconstructed_left = reconstructed_left_side[i]
        reconstructed_right = reconstructed_right_side[i]
        for n in range(N):
            dist_left_side = np.linalg.norm(imgpoints_left[n] -
                                            reconstructed_left[n])
            error_vector[counter] = dist_left_side
            counter += 1

            dist_right_side = np.linalg.norm(imgpoints_right[n] -
                                             reconstructed_right[n])
            error_vector[counter] = dist_right_side
            counter += 1

    return error_vector


if __name__ == "__main__":
    plt.close("all")
    plotter = PlotContainer()
    plotted_img = 0
    breaking_no = 100
    no_fvals = 5000
    exit_flags = ["Iteration limit", "gtol", "ftol", "xtol"]

    # create target pattern and image container
    target_pattern = Checkerboard(8, 11, 15)

    imgcon = ImageContainer("testimgs5/good", "*.tif")
    img_size = imgcon.imgsize
    imgcon.extract(target_pattern)
    no_imgs_considered = 1

    # initialize objects from initial guesses (easily observed parameters)
    theta_in = np.pi - np.pi / 4
    theta_ou = np.pi - 52 * np.pi / 180
    phi_left = np.pi
    phi_right = 0
    dist_bw_mirrors = 40
    dist_to_lens = 15
    mx, my = [3 / 1000] * 2
    focalz = 2.2
    focalx, focaly = 0, 0

    # find r parameter of the mirrors
    left_inner_fixed = Mirror(theta_in, phi_left, 0)
    left_inner_fixed.origin = np.array([0, 0, dist_to_lens + focalz])
    line_left_inner = Line(np.array([0, 0, 0]), left_inner_fixed.orientation)
    point_left_inner = left_inner_fixed.intersect(line_left_inner)
    r_left_inner = point_left_inner[0] / left_inner_fixed.orientation[0]

    left_outer_fixed = Mirror(theta_ou, phi_left, 0)
    left_outer_fixed.origin = np.array([-dist_bw_mirrors / 2, 0, dist_to_lens + focalz])
    line_left_outer = Line(np.array([0, 0, 0]), left_outer_fixed.orientation)
    point_left_outer = left_outer_fixed.intersect(line_left_outer)
    r_left_outer = point_left_outer[0] / left_outer_fixed.orientation[0]

    right_inner_fixed = Mirror(theta_in, phi_right, 0)
    right_inner_fixed.origin = np.array([0, 0, dist_to_lens + focalz])
    line_right_inner = Line(np.array([0, 0, 0]), right_inner_fixed.orientation)
    point_right_inner = right_inner_fixed.intersect(line_right_inner)
    r_right_inner = point_right_inner[0] / right_inner_fixed.orientation[0]

    right_outer_fixed = Mirror(theta_ou, phi_right, 0)
    right_outer_fixed.origin = np.array([dist_bw_mirrors / 2, 0, dist_to_lens + focalz])
    line_right_outer = Line(np.array([0, 0, 0]), right_outer_fixed.orientation)
    point_right_outer = right_outer_fixed.intersect(line_right_outer)
    r_right_outer = point_right_outer[0] / right_outer_fixed.orientation[0]

    tx, ty, tz = 30.0, -40.0, 700.0
    rx, ry, rz = 0.0, 0.0, -np.pi / 2
    rot = R.from_euler('xyz', np.array([rx, ry, rz]))
    [r1, r2, r3] = rot.as_rotvec()

    # create objects of the optical path
    cam = Camera(img_size, mx, my, focalx, focaly, focalz)
    left_inner = Mirror(theta_in, phi_left, r_left_inner)
    left_outer = Mirror(theta_ou, phi_left, r_left_outer)
    right_inner = Mirror(theta_in, phi_right, r_right_inner)
    right_outer = Mirror(theta_ou, phi_right, r_right_outer)
    syst = System(cam, left_inner, left_outer, right_inner, right_outer)

    system_parameters = np.array([theta_in, phi_left, r_left_inner,
                                  theta_ou, phi_left, r_left_outer,
                                  theta_in, phi_right, r_right_inner,
                                  theta_ou, phi_right, r_right_outer,
                                  focalx, focaly, focalz,
                                  mx, my])

    system_bounds = np.array([(-np.inf, np.inf)] * 17)

    initial_target_parameters = np.array([tx, ty, tz, r1, r2, r3])
    df = pd.read_csv(Path("csvs/parameter_names.csv"), header=None)
    df.rename(columns={0: "Parameters"}, inplace=True)
    df["Initial guess"] = np.hstack((system_parameters, initial_target_parameters))

    costs = []
    for num_imgs in range(4, 5):
        no_imgs_considered = num_imgs + 1
        print(f"num imgs: {no_imgs_considered}")

        # create target objects for each image
        targetlist = []
        for i in range(no_imgs_considered):
            targetlist.append(TargetSystem(tx, ty, tz, r1, r2, r3))
        targets = TargetContainer(targetlist)

        target_parameters = np.array([tx, ty, tz, r1, r2, r3] * no_imgs_considered)

        parameters = np.hstack((system_parameters, target_parameters))

        projections_left = []
        projections_right = []
        for i in range(no_imgs_considered):
            projection_left, projection_right = reconstruct_image(syst, targets.tlst[i], imgcon.objpoints)
            projections_left.append(projection_left)
            projections_right.append(projection_right)

        # making sparse matrix for Jacobian
        points_per_image = 2 * target_pattern.gridpoints.shape[0]
        tot_points = no_imgs_considered * points_per_image
        half_points = int(tot_points / 2)
        no_parameters = len(parameters)

        left_mirror_set = set(range(6))
        right_mirror_set = set(range(6, 12))
        shared_set = set(range(12, 17))
        rt_set = set(range(17, no_parameters))

        indices = np.empty(17 * tot_points, dtype=int)
        data = np.ones(17 * tot_points, dtype=np.bool_)
        indptrs = np.zeros(no_parameters + 1, dtype=int)

        index_counter = 0
        rt_counter = 0
        img_counter = 0

        for i in range(no_parameters):
            if i in left_mirror_set:
                indptrs[i + 1] = indptrs[i] + half_points
                for j in range(half_points):
                    indices[index_counter] = 2 * j + 1
                    index_counter += 1
            elif i in right_mirror_set:
                indptrs[i + 1] = indptrs[i] + half_points
                for j in range(half_points):
                    indices[index_counter] = 2 * j
                    index_counter += 1
            elif i in shared_set:
                indptrs[i + 1] = indptrs[i] + tot_points
                for j in range(tot_points):
                    indices[index_counter] = j
                    index_counter += 1
            elif i in rt_set:
                indptrs[i + 1] = indptrs[i] + points_per_image
                for j in range(points_per_image):
                    indices[index_counter] = j + img_counter * points_per_image
                    index_counter += 1
                rt_counter += 1
                if rt_counter % 6 == 0:
                    img_counter += 1
            else:
                raise ValueError("Parameter range is incorrect.")
        mtx = csc_matrix((data, indices, indptrs),
                         shape=(tot_points, no_parameters))

        RTbounds = np.vstack(([(-np.inf, np.inf)] * 2,
                             (600, 800),  # Translations to target in z direction.
                             [(-np.inf, np.inf)] * 3))  # Rotations to target.
        RTbounds = np.vstack(([RTbounds] * no_imgs_considered))
        xbounds = np.vstack(([(-np.inf, np.inf)]*17, RTbounds)).T
        xtuple = tuple(xbounds)
        res = least_squares(objfun, parameters, method='trf', verbose=1, max_nfev=no_fvals, xtol=1e-15, ftol=1e-8,
                            x_scale='jac', jac_sparsity=mtx, bounds=xtuple, args=(syst, targets, imgcon))
