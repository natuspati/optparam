# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 16:42:21 2021

@author: bekdulnm
"""
import numpy as np
import matplotlib.pyplot as plt
from class_defs import *


def reconstruct(point_left, point_right,cam1,left_inner,left_outer,right_inner,right_outer, ax):
    # find left optical path
    line_left = cam1.create_ray(point_left) 
    line_left1 = left_inner.reflect_ray(line_left)
    line_left2 = left_outer.reflect_ray(line_left1)

    # find right optical path
    line_right = cam1.create_ray(point_right)
    line_right1 = right_inner.reflect_ray(line_right)
    line_right2 = right_outer.reflect_ray(line_right1)
    
    
    
    
    print("line left points")
    print(cam1.pixel_to_coordinate(point_left))
    print(line_left1.origin)
    print(line_left2.origin)
    
    print("\n line right points")
    print(cam1.pixel_to_coordinate(point_right))
    print(line_right1.origin)
    print(line_right2.origin)
    
    # find intersection between left and right optical paths
    point3d, error, cleft, cright = line_right2.intersect_ray(line_left2)
    
    print("\n intersection points")
    print(point3d)
    print("error")
    print(error)
    
    # Define the rat in form of xyz to points of interest.
    left = np.vstack((cam1.pixel_to_coordinate(point_left),
                    line_left1.origin,
                    line_left2.origin,
                    cleft)).T
    right = np.vstack((cam1.pixel_to_coordinate(point_right),
                    line_right1.origin,
                    line_right2.origin,
                    cright)).T
    
    return  point3d, left, right

img_size = (1536, 2048)

# # Initialize from guesses.
# phi_in = np.pi/4
# phi_ou = 52*np.pi/180
# dist_bw_mirrors = 40
# dist_to_lens = 15
# mx, my = [3e-3]*2

# focal = 4.4
# focalx, focaly = 0, 0

# theta_in = np.pi - phi_in
# theta_ou = np.pi - phi_ou
# phi_left = np.pi
# phi_right = 0

# left_inner_fixed = Mirror(theta_in, phi_left, 0)
# left_inner_fixed.origin = np.array([0, 0, dist_to_lens + focal])
# line_left_inner = Line(np.array([0,0,0]), left_inner_fixed.orientation)
# point_left_inner = left_inner_fixed.intersect(line_left_inner)
# r_left_inner = point_left_inner[0]/left_inner_fixed.orientation[0]

# left_outer_fixed = Mirror(theta_ou, phi_left, 0)
# left_outer_fixed.origin = np.array([-dist_bw_mirrors/2, 0, dist_to_lens + focal])
# line_left_outer = Line(np.array([0,0,0]), left_outer_fixed.orientation)
# point_left_outer = left_outer_fixed.intersect(line_left_outer)
# r_left_outer = point_left_outer[0]/left_outer_fixed.orientation[0]

# right_inner_fixed = Mirror(theta_in, phi_right, 0)
# right_inner_fixed.origin = np.array([0, 0, dist_to_lens + focal])
# line_right_inner = Line(np.array([0,0,0]), right_inner_fixed.orientation)
# point_right_inner = right_inner_fixed.intersect(line_right_inner)
# r_right_inner = point_right_inner[0]/right_inner_fixed.orientation[0]

# right_outer_fixed = Mirror(theta_ou, phi_right, 0)
# right_outer_fixed.origin = np.array([dist_bw_mirrors/2, 0, dist_to_lens + focal])
# line_right_outer = Line(np.array([0,0,0]), right_outer_fixed.orientation)
# point_right_outer = right_outer_fixed.intersect(line_right_outer)
# r_right_outer = point_right_outer[0]/right_outer_fixed.orientation[0]


# Initialize from optmization result.
[theta_in, phi_left, r_left_inner,
          theta_ou, phi_left, r_left_outer,
          theta_in, phi_right, r_right_inner,
          theta_ou, phi_right, r_right_outer,
          focalx, focaly, focal,
          mx, my,
          tx, ty, tz,
          rx, ry, rz] = [2.356194490192344837e+00,
3.141592653591709805e+00,
-1.354396381167727270e+01,
2.234021442552749370e+00,
3.062424395167283375e+00,
3.604572618745570445e+00,
2.351693828744827464e+00,
9.999306359978554666e-11,
-1.392641040624883608e+01,
2.234465342032864399e+00,
1.715449055494872727e-02,
4.097682358497122479e+00,
-2.846060504110957488e-03,
-1.324820520811906266e-02,
4.382672860134926296e+00,
3.010031049892035633e-03,
3.514739545969662940e-03,
-4.556594639930304425e+01,
8.327803977896034837e+01,
5.032455032897570959e+02,
1.887639070186377666e-01,
-9.847899814834380761e-02,
-1.656392036518039790e+00]

# [2.356194490192344837e+00,
# 3.141592653589793116e+00,
# -1.371787155501901978e+01,
# 2.234021442552741821e+00,
# 3.141592653589793116e+00,
# 3.816382450816670158e+00,
# 2.356194490192344837e+00,
# 0.000000000000000000e+00,
# -1.371787155501901978e+01,
# 2.234021442552741821e+00,
# 0.000000000000000000e+00,
# 3.816382450816670158e+00,
# 0.000000000000000000e+00,
# 0.000000000000000000e+00,
# 4.400000000000000355e+00,
# 3.000000000000000062e-03,
# 3.000000000000000062e-03,
# -4.000000000000000000e+01,
# 8.000000000000000000e+01,
# 4.780000000000000000e+02,
# 0.000000000000000000e+00,
# 0.000000000000000000e+00,
# -1.570796326794896558e+00]


# [2.356194490192344837e+00,
# 3.141592653591709805e+00,
# -1.354396381167727270e+01,
# 2.234021442552749370e+00,
# 3.062424395167283375e+00,
# 3.604572618745570445e+00,
# 2.351693828744827464e+00,
# 9.999306359978554666e-11,
# -1.392641040624883608e+01,
# 2.234465342032864399e+00,
# 1.715449055494872727e-02,
# 4.097682358497122479e+00,
# -2.846060504110957488e-03,
# -1.324820520811906266e-02,
# 4.382672860134926296e+00,
# 3.010031049892035633e-03,
# 3.514739545969662940e-03,
# -4.556594639930304425e+01,
# 8.327803977896034837e+01,
# 5.032455032897570959e+02,
# 1.887639070186377666e-01,
# -9.847899814834380761e-02,
# -1.656392036518039790e+00]

cam1 = Camera(img_size, mx, my, focalx, focaly, focal)

left_inner = Mirror(theta_in, phi_left, r_left_inner)

left_outer = Mirror(theta_ou, phi_left, r_left_outer)

right_inner = Mirror(theta_in, phi_right, r_right_inner)

right_outer = Mirror(theta_ou, phi_right, r_right_outer)


color = ['C0', 'C1','C2','C3']


# Make the figure.
plt.figure(1)
ax = plt.axes(projection='3d')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

sensor_left = np.array([[img_size[1]/2,0],
                        [img_size[1],0],
                        [img_size[1],img_size[0]],
                        [img_size[1]/2,img_size[0]]])  
sensor_right = np.array([[0,0],
                        [img_size[1]/2,0],
                        [img_size[1]/2,img_size[0]],
                        [0,img_size[0]]])  
left_inner_mirror_pos = np.zeros((3, 5))
left_outer_mirror_pos = np.zeros((3, 5))
right_inner_mirror_pos = np.zeros((3, 5))
right_outer_mirror_pos = np.zeros((3, 5))

# loop over all corners
for num in range(4):
    p, left, right = reconstruct(sensor_left[num], sensor_right[num],cam1,left_inner,left_outer,right_inner,right_outer, ax)
    left_inner_mirror_pos[:, num] = left[:, 1]
    left_outer_mirror_pos[:, num] = left[:, 2]
    right_inner_mirror_pos[:, num] = right[:, 1]
    right_outer_mirror_pos[:, num] = right[:, 2]
left_inner_mirror_pos[:, -1] = left_inner_mirror_pos[:, 0]
left_outer_mirror_pos[:, -1] = left_outer_mirror_pos[:, 0]
right_inner_mirror_pos[:, -1] = right_inner_mirror_pos[:, 0]
right_outer_mirror_pos[:, -1] = right_outer_mirror_pos[:,0]

plt.plot(*left_inner_mirror_pos, c="C1")
plt.plot(*left_outer_mirror_pos, c="C1")
plt.plot(*right_inner_mirror_pos, c="C3")
plt.plot(*right_outer_mirror_pos, c="C3")


# Plat the rays of a point.
point_left = np.array([1415.62,582.883])
point_right = np.array([847.908,550.651])

p, left, right = reconstruct(point_left, point_right,cam1,left_inner,left_outer,right_inner,right_outer, ax)
plt.plot(*left,c="C0")
plt.plot(*right,c="C2")
ax.scatter(p[0],p[1],p[2],c="C5")
