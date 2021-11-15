# -*- coding: utf-8 -*-
"""
Created on Sun Oct 24 15:45:02 2021

@author: bekdulnm
"""


# imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from scipy.optimize import least_squares
import scipy.optimize._lsq.trf as trf
trf.passlist = []

from classes import *
from system import *
from targets import *


def objfun(param, imgpoints_left, imgpoints_right, objpoints, img_size):
    """
    Objective function,
    
    The function computes the error vector for the images provided.

    Parameters
    ----------
    param : array_like
        List of parameters. Size depends on number of images: x_k = 17 + 6*I,
        where x_k - size of param, I - number of images.
    imgpoints_left : ndarray
        Extracted points for the left optical path of all images.
    imgpoints_right : ndarray
        Extracted points for the right optical path of all images.
    objpoints : ndarray
        Object points of size (N, 3), where N- number of points per image.
    img_size : tuple
        Resolution of an image.

    Returns
    -------
    error : TYPE
        DESCRIPTION.

    """
    # Expand the parameter vector and the consecutive objects.
    [theta_in, phi_left, r_left_inner,
              theta_ou, phi_left, r_left_outer,
              theta_in, phi_right, r_right_inner,
              theta_ou, phi_right, r_right_outer,
              focalx, focaly, focal,
              mx, my,
              tx, ty, tz,
              rx, ry, rz] = param
    
    cam1 = Camera(img_size, mx, my, focalx, focaly, focal)

    left_inner = Mirror(theta_in, phi_left, r_left_inner)

    left_outer = Mirror(theta_ou, phi_left, r_left_outer)
    
    right_inner = Mirror(theta_in, phi_right, r_right_inner)
    
    right_outer = Mirror(theta_ou, phi_right, r_right_outer)
    
    target = Target(tx, ty, tz, rx, ry, rz)
    
    # Iterate over all points
    
    reconstructed_points = np.zeros((70, 3))
    error = np.zeros(70)
    midpointerror = []
    for index, point_left in enumerate(imgpoints_left):
        point_right = imgpoints_right[index]
        reconstructed_points[index], mperror = reconstruct(point_left, point_right,
                                                  cam1,
                                                  left_inner,
                                                  left_outer,
                                                  right_inner,
                                                  right_outer,
                                                  target)
        midpointerror.append(mperror)
        error[index] = np.linalg.norm(objpoints[index]-reconstructed_points[index])
    return error


def reconstruct_image(system, target, objpoints):
    reconstructed_points = np.zeros((objpoints.shape[0], 4))
    for i, objpoint in enumerate(objpoints):
        reconstructed_points[i] = reconstruct_point(system, target, objpoint)
    return reconstructed_points

def reconstruct_point(system, target, objpoint):  
    # find left optical path
    line_left = target.create_ray(objpoint, system.focal_left)
    line_left1 = left_outer.reflect_ray(line_left)
    line_left2 = left_inner.reflect_ray(line_left1)

    # find right optical path
    line_right = target.create_ray(objpoint, system.focal_right)
    line_right1 = right_outer.reflect_ray(line_right)
    line_right2 = right_inner.reflect_ray(line_right1)
    
    # find intersection between left/right optical paths with sensor
    projection_left = system.cam.intersect_ray(line_left2)
    projection_right = system.cam.intersect_ray(line_right2)
    return np.hstack((projection_left, projection_right))

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
    mx, my = [3e-3]*2
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

    # define initial translations and rotations towards the target system
    tx, ty, tz = -40.0, 80.0, 478.0
    rx, ry, rz = 0.0, 0.0, -np.pi/2
    
    rot = R.from_euler('xyz', np.array([rx, ry, rz]))
    [r1, r2, r3] = rot.as_rotvec()

    # create objects of the optical path
    cam = Camera(img_size, mx, my, focalx, focaly, focalz)

    left_inner = Mirror(theta_in, phi_left, r_left_inner)

    left_outer = Mirror(theta_ou, phi_left, r_left_outer)

    right_inner = Mirror(theta_in, phi_right, r_right_inner)

    right_outer = Mirror(theta_ou, phi_right, r_right_outer)
    
    no_imgs_considered = 2
    targetlist = []
    for i in range(no_imgs_considered):
        targetlist.append(TargetSystem(tx, ty, tz, r1, r2, r3))

    sys = System(cam, left_inner, left_outer, right_inner, right_outer)
    sys.find_reflected_focals()
    targets = TargetContainer(targetlist)

    projections = []
    for i in range(no_imgs_considered):
        projections_per_image = reconstruct_image(sys,
                                                  targets.tlst[i],
                                                  imgcon.objpoints)
        projections.append(projections_per_image)

    plt.close("all")
    img1 = plt.imread(imgcon.stereoimgs[0])
    plt.imshow(img1)
    projections1 = projections[0]
    xs = np.hstack((projections1[:,0], projections1[:,2]))
    ys = np.hstack((projections1[:,1], projections1[:,3]))
    plt.scatter(xs, ys, s=80, facecolors='none', edgecolors='r')


    # # argument list
    # ivs = np.array([theta_in, phi_left, r_left_inner,
    #                 theta_ou, phi_left, r_left_outer,
    #                 theta_in, phi_right, r_right_inner,
    #                 theta_ou, phi_right, r_right_outer,
    #                 focalx, focaly, focalz,
    #                 mx, my,
    #                 tx, ty, tz,
    #                 r1, r2, r3])

    
    
    
    
    
    
    # plt.close("all")
    # # plot extracted feature points
    # f2 = plt.figure(1)
    # plt.imshow(ii)    
    # plt.scatter(imgpoints_left[:,0], imgpoints_left[:,1],s=80, facecolors='none', edgecolors='r')
    # plt.scatter(imgpoints_right[:,0], imgpoints_right[:,1], s=80, facecolors='none', edgecolors='r')
    
    # # plotting reconstructed points
    # f1 = plt.figure(2)    
    # ax = plt.axes(projection='3d')
    # ax.scatter(objpoints[:,0],objpoints[:,1],objpoints[:,2], s=80, color='C0') #facecolors='none', edgecolors='C1')
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
   
    
    # diff = []
    # for i, point in enumerate(objpoints):
    #     if i % 7 == 0:
    #         for j in range(i,i+6):
    #             diff.append(np.linalg.norm(objpoints[j] - objpoints[j+1]))
                
    # for i in range(0, 63, 7):
    #     for j in range(7):
    #         diff.append(np.linalg.norm(objpoints[i+j] - objpoints[i+j+7]))            

    # diff = np.array(diff)
    # print(f"average distance bw neighbouring points: {np.average(diff)}")
    
    
    # xbounds = np.vstack(([(0, np.pi),(0,2*np.pi),(-np.inf,np.inf)]*4,
    #                    [(-np.inf,np.inf)]*11))
    # xbounds = xbounds.T
    # xtuple = tuple(xbounds)
        
    # res1 = least_squares(objfun,
    #               ivs,
    #               method='trf',
    #               verbose=2,
    #               x_scale='jac',
    #               bounds=xtuple,
    #               max_nfev=1000,
    #               args=(imgpoints_left, imgpoints_right, objpoints, img_size)
    


