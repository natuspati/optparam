# -*- coding: utf-8 -*-
"""
Created on Sun Oct 24 15:45:02 2021

@author: bekdulnm
"""


# imports
from  sys import getsizeof
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from scipy.optimize import least_squares
from scipy.sparse import csc_matrix
import scipy.optimize._lsq.trf as trf
trf.passlist = []

from classes import *
from system import *
from targets import *


def objfun(parameters, sys1tem, targets, imgcon):
    objpoints = imgcon.objpoints
    no_imgs = len(targets.tlst)
    no_points = len(imgcon.objpoints)
    
    sys1tem_parameters = parameters[:17]
    sys1tem.update(sys1tem_parameters)
    target_parameters =  np.reshape(parameters[17:], (no_imgs, 6)) #check
    targets.update(target_parameters)

    reconstructed_impgpoints_left = []
    reconstructed_impgpoints_right = []

    for trgt in targets.tlst:
        reconstructed_left, reconstructed_right = reconstruct_image(sys1tem,
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


def reconstruct_image(sys1tem, target, objpoints):
    reconstructed_left = np.zeros((objpoints.shape[0], 2))
    reconstructed_right = np.zeros((objpoints.shape[0], 2))
    for i, objpoint in enumerate(objpoints):
        reconstructed_left[i], reconstructed_right[i] = \
            reconstruct_point(sys1tem, target, objpoint)
    return reconstructed_left, reconstructed_right

def reconstruct_point(sys1tem, target, objpoint):  
    # find left optical path
    line_left = target.create_ray(objpoint, sys1tem.focal_left)
    line_left1 = left_outer.reflect_ray(line_left)
    line_left2 = left_inner.reflect_ray(line_left1)

    # find right optical path
    line_right = target.create_ray(objpoint, sys1tem.focal_right)
    line_right1 = right_outer.reflect_ray(line_right)
    line_right2 = right_inner.reflect_ray(line_right1)
    
    # find intersection between left/right optical paths with sensor
    projection_left = sys1tem.cam.intersect_ray(line_left2)
    projection_right = sys1tem.cam.intersect_ray(line_right2)
    return projection_left, projection_right
    # return np.hstack((projection_left, projection_right))

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
    # create checkerboard and image container
    cb = Checkerboard(8, 11, 15)
    imgcon = ImageContainer("testimgs")
    img_size = imgcon.imgsize
    imgcon.extract(cb)
    
    # initialize objects from initial guesses
    # easily observed parameters
    phi_in = np.pi/4
    phi_ou = 52*np.pi/180
    dist_bw_mirrors = 40
    dist_to_lens = 15
    mx, my = [2.82/10000]*2
    focalz = 4.4
    focalx, focaly = 0, 0
    
    # define mirrors: theta angles are with z axis, phi with x axis
    theta_in = np.pi - phi_in
    theta_ou = np.pi - phi_ou
    phi_left = np.pi
    phi_right = 0
    
    # find r paramter of the mirrors
    left_inner_fixed = Mirror(theta_in, phi_left, 0)
    left_inner_fixed.origin = np.array([0, 0, dist_to_lens + focalz])
    line_left_inner = Line(np.array([0,0,0]), left_inner_fixed.orientation)
    point_left_inner = left_inner_fixed.intersect(line_left_inner)
    r_left_inner = point_left_inner[0]/left_inner_fixed.orientation[0]
    
    left_outer_fixed = Mirror(theta_ou, phi_left, 0)
    left_outer_fixed.origin = np.array([-dist_bw_mirrors/2, 0, dist_to_lens + focalz])
    line_left_outer = Line(np.array([0,0,0]), left_outer_fixed.orientation)
    point_left_outer = left_outer_fixed.intersect(line_left_outer)
    r_left_outer = point_left_outer[0]/left_outer_fixed.orientation[0]
    
    right_inner_fixed = Mirror(theta_in, phi_right, 0)
    right_inner_fixed.origin = np.array([0, 0, dist_to_lens + focalz])
    line_right_inner = Line(np.array([0,0,0]), right_inner_fixed.orientation)
    point_right_inner = right_inner_fixed.intersect(line_right_inner)
    r_right_inner = point_right_inner[0]/right_inner_fixed.orientation[0]
    
    right_outer_fixed = Mirror(theta_ou, phi_right, 0)
    right_outer_fixed.origin = np.array([dist_bw_mirrors/2, 0, dist_to_lens + focalz])
    line_right_outer = Line(np.array([0,0,0]), right_outer_fixed.orientation)
    point_right_outer = right_outer_fixed.intersect(line_right_outer)
    r_right_outer = point_right_outer[0]/right_outer_fixed.orientation[0]

    # define initial translations and rotations towards the target sys1tem
    tx, ty, tz = 0.0, 0.0, 600.0
    rx, ry, rz = 0.0, 0.0, -np.pi/2
    
    rot = R.from_euler('xyz', np.array([rx, ry, rz]))
    [r1, r2, r3] = rot.as_rotvec()

    # create objects of the optical path
    cam = Camera(img_size, mx, my, focalx, focaly, focalz)
    left_inner = Mirror(theta_in, phi_left, r_left_inner)
    left_outer = Mirror(theta_ou, phi_left, r_left_outer)
    right_inner = Mirror(theta_in, phi_right, r_right_inner)
    right_outer = Mirror(theta_ou, phi_right, r_right_outer)
    
    no_imgs_considered = 1
    targetlist = []
    for i in range(no_imgs_considered):
        targetlist.append(TargetSystem(tx, ty, tz, r1, r2, r3))

    sys1 = System(cam, left_inner, left_outer, right_inner, right_outer)
    targets = TargetContainer(targetlist)

    projections_left = []
    projections_right = []
    for i in range(no_imgs_considered):
        projection_left, projection_right = reconstruct_image(sys1,
                                                              targets.tlst[i],
                                                              imgcon.objpoints)
        projections_left.append(projection_left)
        projections_right.append(projection_right)

    sys1tem_parameters = np.array([theta_in, phi_left, r_left_inner,
                                  theta_ou, phi_left, r_left_outer,
                                  theta_in, phi_right, r_right_inner,
                                  theta_ou, phi_right, r_right_outer,
                                  focalx, focaly, focalz,
                                  mx, my])

    target_parameters = np.array([tx, ty, tz, r1, r2, r3] * no_imgs_considered)

    parameters = np.hstack((sys1tem_parameters, target_parameters))
    evec = objfun(parameters, sys1, targets, imgcon)

    points_per_image = 2 * cb.gridpoints.shape[0]
    tot_points = no_imgs_considered * points_per_image
    half_points = int(tot_points / 2)
    no_parameters = len(parameters)

    left_mirror_set = set(range(6))
    right_mirror_set = set(range(6,12))
    shared_set = set(range(12,17))
    rt_set = set(range(17,no_parameters))

    indices = np.empty(17*tot_points, dtype=int)
    data = np.ones(17*tot_points, dtype=np.bool_)
    indptrs = np.zeros(no_parameters+1, dtype=int)

    index_counter = 0
    rt_counter = 0
    img_counter = 0

    for i in range(no_parameters):
        if i in left_mirror_set:
            indptrs[i+1] = indptrs[i] + half_points
            for j in range(half_points):
                indices[index_counter] = 2*j + 1
                index_counter += 1
        elif i in right_mirror_set:
            indptrs[i+1] = indptrs[i] + half_points
            for j in range(half_points):
                indices[index_counter] = 2*j
                index_counter += 1
        elif i in shared_set:
            indptrs[i+1] = indptrs[i] + tot_points
            for j in range(tot_points):
                indices[index_counter] = j
                index_counter += 1
        elif i in rt_set:
            indptrs[i+1] = indptrs[i] + points_per_image
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
    
    res1 = least_squares(objfun,
                          parameters,
                          method='trf',
                          jac_sparsity=mtx,
                          x_scale='jac',
                          verbose=2,
                          max_nfev=100,
                          xtol=1e-15,
                          ftol=1e-8,
                          args=(sys1, targets, imgcon))

    # reconstructing after optimization
    optimized_left = []
    optimized_right = []
    for i in range(no_imgs_considered):
        projection_left, projection_right = reconstruct_image(sys1,
                                                              targets.tlst[i],
                                                              imgcon.objpoints)
        optimized_left.append(projection_left)
        optimized_right.append(projection_right)

    # a = trf.passlist
    # np.savetxt('forplotting1.csv', a, delimiter=',')
    
    # plt.close("all")
    # plt.bar(np.arange(len(a[-1])), a[-1])
    # ax = plt.gca()
    # ax.set_yscale('log')
    # plt.xlabel('parameter', fontsize=16)
    # plt.ylabel('gradient value', fontsize=16)
    # plt.yticks(fontsize=16)
    # plt.xticks(ticks=np.arange(len(a[-1])),
    #            labels=['theta_in', 'phi_left', 'r_left_inner',
    #                    'theta_ou', 'phi_left', 'r_left_outer',
    #                    'theta_in', 'phi_right', 'r_right_inner',
    #                    'theta_ou', 'phi_right', 'r_right_outer',
    #                    'focalx', 'focaly', 'focalz',
    #                    'mx', 'my',
    #                    'Tx', 'Ty', 'Tz',
    #                    'r1', 'r2', 'r3'], rotation=90, fontsize=16)
    
    
    
    plt.close("all")
    img1 = plt.imread(imgcon.stereoimgs[0])
    plt.imshow(img1)
    
    # rpoints_left = projections_left[0]
    # rpoints_right = projections_right[0]
    # xs = np.hstack((rpoints_left[:,0], rpoints_right[:,0]))
    # ys = np.hstack((rpoints_left[:,1], rpoints_right[:,1]))
    # plt.scatter(xs, ys, s=100, facecolors='none', edgecolors='b') 

    opoints_left = optimized_left[0]
    opoints_right = optimized_right[0]
    xs = np.hstack((opoints_left[:,0], opoints_right[:,0]))
    ys = np.hstack((opoints_left[:,1], opoints_right[:,1]))
    plt.scatter(xs, ys, s=100, facecolors='none', edgecolors='g')

    ipoints_left = imgcon.imgpoints_left[0].reshape(70,2)
    ipoints_right = imgcon.imgpoints_right[0].reshape(70,2)
    xs = np.hstack((ipoints_left[:,0], ipoints_right[:,0]))
    ys = np.hstack((ipoints_left[:,1], ipoints_right[:,1]))
    plt.scatter(xs, ys, s=100, facecolors='none', edgecolors='r') 
    plt.show()
