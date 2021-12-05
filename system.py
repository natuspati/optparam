# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 13:46:35 2021

@author: bekdulnm
"""

import numpy as np
from scipy.spatial.transform import Rotation as R

from classes import Line, ImageContainer


class System(object):
    def __init__(self, cam, left_inner, left_outer,
                 right_inner, right_outer):
        self.cam = cam
        self.left_inner = left_inner
        self.left_outer = left_outer
        self.right_inner = right_inner
        self.right_outer = right_outer
        self.focal_left = np.array([])
        self.focal_right = np.array([])
        self._find_reflected_focals()

    # def update(self, system_parameters):
    #     theta_left_in, phi_left_in, r_left_in,\
    #     theta_left_ou, phi_left_ou, r_left_ou,\
    #     theta_right_in, phi_right_in, r_right_in,\
    #     theta_right_ou, phi_right_ou, r_right_ou,\
    #     focalx, focaly, focalz = system_parameters

    #     self.cam.update(focalx, focaly, focalz)
    #     self.left_inner.update(theta_left_in, phi_left_in, r_left_in)
    #     self.left_outer.update(theta_left_ou, phi_left_ou, r_left_ou)
    #     self.right_inner.update(theta_right_in, phi_right_in, r_right_in)
    #     self.right_outer.update(theta_right_ou, phi_right_ou, r_right_ou)

    #     self.find_reflected_focals()    

    def update(self, system_parameters):
        theta_left_in, phi_left_in, r_left_in, \
        theta_left_ou, phi_left_ou, r_left_ou, \
        theta_right_in, phi_right_in, r_right_in, \
        theta_right_ou, phi_right_ou, r_right_ou, \
        focalx, focaly, focalz, \
        mx, my = system_parameters

        self.cam.update(mx, my, focalx, focaly, focalz)
        self.left_inner.update(theta_left_in, phi_left_in, r_left_in)
        self.left_outer.update(theta_left_ou, phi_left_ou, r_left_ou)
        self.right_inner.update(theta_right_in, phi_right_in, r_right_in)
        self.right_outer.update(theta_right_ou, phi_right_ou, r_right_ou)

        self._find_reflected_focals()

    def _find_reflected_focals(self):
        focal = self.cam.translations_lens

        focal_left_in = self.left_inner.reflect_point(focal)
        focal_left_ou = self.left_outer.reflect_point(focal_left_in)
        self.focal_left = focal_left_ou

        focal_right_in = self.right_inner.reflect_point(focal)
        focal_right_ou = self.right_outer.reflect_point(focal_right_in)
        self.focal_right = focal_right_ou


class TargetContainer(object):
    def __init__(self, tlst):
        self.tlst = tlst

    def update(self, target_parameters):
        for i in range(len(self.tlst)):
            self.tlst[i].update(*target_parameters[i])


class Camera(object):
    """
    Camera object.
    
    Camera that describes intrinsic parameters of the camera module.
    
    Parameters
    ----------
    img_size : tuple
        Image resolution.
    mx : float
        Scaling of a pixel on the sensor in x direction.
    my : float
        Scaling of a pixel on the sensor in y direction.
    focalx : float
        Focal length of the camera in $x$ direction.
    focaly : float
        Focal length of the camera in $y$ direction.
    focalz : float
        Focal length of the camera in $z$ direction.
    
    Attribute
    ---------
    mx : float
        Scaling of a pixel on the sensor in x direction.
    my : float
        Scaling of a pixel on the sensor in y direction.
    u0 : float
        Centre of columns of the image in pixels.
    v0 : float
        Centre of rows of the image in pixels.
    translations_lens : ndarray
        Translation vector between lens and centre of the sensor.
    """

    def __init__(self, img_size, mx, my, focalx, focaly, focalz):
        self.mx = 10 * mx
        self.my = 10 * my
        self.u0 = img_size[1] / 2
        self.v0 = img_size[0] / 2
        self.translations_lens = np.array([focalx, focaly, focalz])

    # def update(self, focalx, focaly, focalz):
    #     self.translations_lens = np.array([focalx, focaly, focalz])

    def update(self, mx, my, focalx, focaly, focalz):
        self.mx = 10 * mx
        self.my = 10 * my
        self.translations_lens = np.array([focalx, focaly, focalz])

    def pixel_to_coordinate(self, pixel_position):
        """
        Pixel to mm converter.
        
        Convert pixel posistion to a physcial position in space.

        Parameters
        ----------
        pixel_position : ndarray
            Position of a point in pixel coordinates.

        Returns
        -------
        coordinate : ndarray
            position of a point in mm coordinates.
        """
        x = (pixel_position[0] - self.u0) * self.mx
        y = (pixel_position[1] - self.v0) * self.my
        coordinate = np.array([x, y, 0])
        return coordinate

    def create_ray(self, pixel_position):
        """
        Ray from a point on a sensor.
        
        Create an outgoing line from a pixel position towards a lens.

        Parameters
        ----------
        pixel_position : array
            Position of a point in pixel coordinates.

        Returns
        -------
        Line
            Line containing position of lens and an outgoing ray.
        """
        sensor_position = self.pixel_to_coordinate(pixel_position)
        ray = self.translations_lens - sensor_position
        return Line(self.translations_lens, ray)

    def intersect_ray(self, ray):
        ret = ray.intersect_point(self.translations_lens)
        if ret is True:
            d = - ray.origin[2] / ray.orientation[2]
            intersection_point = ray.origin + d * ray.orientation
        else:
            raise ValueError("Projection from object point does not intersect the lens")
        projection_point = self.coordinate_to_pixel(intersection_point)
        return projection_point

    def coordinate_to_pixel(self, coordinate):
        if np.isclose(coordinate[2], 0):
            u = coordinate[0] / self.mx + self.u0
            v = coordinate[1] / self.my + self.v0
        else:
            raise ValueError("Projection from object point does not lie on the sensor")
        return np.array([u, v])


class Mirror(object):
    """
    Mirror object.
    
    Mirror is described as an infinite plane wit a normal and point of origin.
    Default initilization uses spherical coordinates [1]_. Internally, a point
    and a normal definition is used and can be initilized from.
    
    Parameters
    ----------
    theta : float
        Angle theta with z axis of the spherical coordinates (theta, phi, r).
    phi : float
        Angle phi on x-y plane of the spherical coordinates (theta, phi, r).
    r : float
        Distance component of the spherical coordinates (theta, phi, r).
    
    Attribute
    ---------
    origin : ndarray
        Any point belonging to the mirror plane.
    orientation : ndarray
        The direction of the normal of the mirror.
       
    .. [1] "Spherical coordinatse", wikipedia
        https://en.wikipedia.org/wiki/Spherical_coordinate_system#Unique_coordinates.
    """

    def __init__(self, theta, phi, r):
        self.orientation = np.array([np.cos(phi) * np.sin(theta),
                                     np.sin(phi) * np.sin(theta),
                                     np.cos(theta)])
        self.origin = r * self.orientation

    @classmethod
    def from_point_normal(cls, origin, orientation):
        """
        Initilize mirror from origin and orientation directly.
        """
        obj = cls.new(cls)
        obj.origin = origin
        obj.orientation = orientation
        return obj

    @property
    def orientation(self):
        return self._orientation

    @orientation.setter
    def orientation(self, orientation):
        self._orientation = orientation / np.linalg.norm(np.array(orientation))

    def update(self, theta, phi, r):
        self.orientation = np.array([np.cos(phi) * np.sin(theta),
                                     np.sin(phi) * np.sin(theta),
                                     np.cos(theta)])
        self.origin = r * self.orientation

    def intersect(self, line):
        """
        Intersection of a mirror with a line.
        
        Implmentation of the intersection of a line and a plane using the 
        algebraic form [1]_

        Parameters
        ----------
        line : Line
            This line is intersected with the plane of mirror.

        Raises
        ------
        ZeroDivisionError
            If the supplied line is parallel to the mirror, no intersection
            point exists.

        Returns
        -------
        p : ndarray
            Intersection point.

        .. [1] "Line-plane intersection", wikipedia
            https://en.wikipedia.org/wiki/Line%E2%80%93plane_intersection#Algebraic_form.
        """

        # Rename variables according to wikipedia's convention
        p0 = self.origin
        n = self.orientation
        l0 = line.origin
        l = line.orientation

        # Check for parallel case
        dot_product = np.dot(l, n)
        if dot_product == 0:
            raise ZeroDivisionError("Line does not intersect with a mirror " +
                                    "plane because they are parallel")
        d = np.dot(p0 - l0, n) / dot_product
        p = l0 + d * l
        return np.array(p)

    def reflect_ray(self, line):
        """
        Reflect incoming line with the mirror 
        
        Implmentation of the intersection of a line and a plane using the 
        algebraic form and the reflection of a line and a plance [1]_.

        Parameters
        ----------
        line : Line
            This line is intersected and reflected with the plane of mirror.
            
        Raises
        ------
        UserWarning
            Reflected ray is orthogonal to the mirror, hence in the reverse
            direction of the incoming ray.

        Returns
        -------
        Line
            Reflected line with origin at the intersection point of mirror and
            incoming line.

        .. [1] "Reflection acrss a line in the plane", wikipedia
            https://en.wikipedia.org/wiki/Reflection_(mathematics)#Reflection_across_a_line_in_the_plane
        """
        # Rename variables according to wikipedia's convention
        v = line.orientation
        a = self.orientation

        # Calculate intersection point.
        intersection_point = self.intersect(line)

        # calculate reflected ray.
        dot_product = np.dot(v, a)
        if np.isclose(abs(dot_product), 1, rtol=1e-8):
            raise UserWarning("Reflected line is a reverse of incoming line " +
                              "because the mirror is perpendicular to the ray.")
        reflected_line_orientation = v - 2 * dot_product / np.dot(a, a) * a
        return Line(intersection_point, reflected_line_orientation)

    def reflect_point(self, point):
        # define normal from point towards plane
        n = Line(point, self.orientation)

        # intersect new line with plane
        proj_point = self.intersect(n)

        # mirror point = 2*(proj_point - point) + point
        mirror_point = proj_point + (proj_point - point)
        return np.array(mirror_point)


class TargetSystem(object):
    """
    Target object.
    
    Target contains rotation matrix and translation vector to convert the
    global coordinates to the local target coordinates.
    
    Parameters
    ----------
    tx : float
        Translation from the local target coordinate to the global sensor
        coordinate system in $x$ direction.
    ty : float
        Translation from the local target coordinate to the global sensor
        coordinate system in $y$ direction.
    tz : float
        Translation from the local target coordinate to the global sensor
        coordinate system in $z$ direction.
    r1 : float
        Rotation vector element.
    r2 : float
        Rotation vector element.
    r3 : float
        Rotation vector element.
    
    Attributes
    ----------
    tranvector : ndarray
        Translation vector from the local target coordinates to the gloval
        sensor coordinates.
    rotmatrix : ndarray
        Rotation matrix from the local target coordinates to the gloval
        sensor coordinates.
    """

    def __init__(self, tx, ty, tz, r1, r2, r3):
        self.tranvector = np.array([tx, ty, tz])
        rot = R.from_rotvec(np.array([r1, r2, r3]))
        self.rotmatrix = rot.as_matrix()

    def update(self, tx, ty, tz, r1, r2, r3):
        self.tranvector = np.array([tx, ty, tz])
        rot = R.from_rotvec(np.array([r1, r2, r3]))
        self.rotmatrix = rot.as_matrix()

    def to_local(self, global_point):
        local_point = np.dot(np.linalg.inv(self.rotmatrix), global_point - self.tranvector)
        return local_point

    def to_global(self, local_point):
        global_point = np.dot(self.rotmatrix, local_point) + self.tranvector
        return global_point

    def create_ray(self, point, focal):
        global_point = self.to_global(point)
        return Line(global_point, focal - global_point)


if __name__ == '__main__':
    print("statement")
