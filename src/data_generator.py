import numpy as np
from scipy.spatial.transform import Rotation
from copy import copy
from extra_data_types import ndarray
from target_types import Checkerboard
from homography import Homography

ANGLE_INCREMENT = np.array([0, 0, 15])
TRANSLATION_INCREMENT = np.array([5, 10, 50])


class DataGenerator:
    def __init__(self, no_images: int,
                 perturb_by: float,
                 verticals: int,
                 horizontals: int,
                 distance: float | int,
                 f_x: float | int,
                 f_y: float | int,
                 c_x: float | int,
                 c_y: float | int,
                 inner_mirror_angle: float | int,
                 outer_mirror_angle: float | int,
                 dist_bw_mirrors: float | int,
                 dist_to_mirrors: float | int,
                 default_angles: ndarray,
                 default_translations: ndarray):
        """
        Data generator for synthetic projections. Provides ideal case parameters upon initialization. Perturbations
        in parameters or projections can be also added. Initialized perturbed values are randomly spread within 10%
        of the ideal case.
        
        Parameters
        ----------
        no_images
            Number of images to be considered.
        perturb_by
            Amount of max shift from ideal case parameters in offset parameters.
        verticals
            Number of vertical edges on the target grid.
        horizontals
            Number of horizontal edges on the target grid.
        distance
            The shortest distance between edges on the target grid.
        f_x
            Focal for x-axis in camera coordinate system.
        f_y
            Focal for y-axis in camera coordinate system.
        c_x
            Principal point in x-axis in camera coordinate system.
        c_y
            Principal point in y-axis in camera coordinate system.
        inner_mirror_angle
            Angle of inner mirrors wrt to z-normal in xz plane.
        outer_mirror_angle
            Angle of outer mirrors wrt to z-normal in xz plane.
        dist_bw_mirrors
            Distance between the bases of inner and outer mirrors.
        dist_to_mirrors
            Distance between the bases of mirrors and camera.
        default_angles
            Euler angles for the initial image.
        default_translations
            Translation vector for the initial image
        """
        # Generate world points from a target type.
        target_type = Checkerboard(verticals, horizontals, distance)
        self.world_points = target_type.points
        self.__number = no_images
        self.perturb_by = perturb_by
        
        # Generate intrinsic matrix
        intrinsic_matrix = np.array([[f_x, 0, c_x],
                                     [0, f_y, c_y],
                                     [0, 0, 1]])
        
        # Generate rotation and translation vectors for all images.
        rotations = np.empty((self.number, 3))
        translations = np.empty((self.number, 3))
        for i in range(self.number):
            angles = default_angles + i * ANGLE_INCREMENT
            rotations_object = Rotation.from_euler('xyz', angles, degrees=True)
            rotations[i] = rotations_object.as_rotvec()
            translations[i] = default_translations + i * TRANSLATION_INCREMENT
        
        # Create a homography object and set the initialized parameters as the ideal case parameters.
        self.__homography = Homography(intrinsic_matrix, inner_mirror_angle, outer_mirror_angle, dist_bw_mirrors,
                                       dist_to_mirrors, rotations, translations)
        self.ideal_parameters = self.homography.parameters
        
        # Project homography from world points to image points using the ideal case parameters.
        self.left_ideal_points, self.right_ideal_points = self.homography.project_to_images(self.world_points)
        
        self.offset_parameters = self.ideal_parameters.copy()
        self.perturb_parameters(perturb_by)
    
    def __str__(self):
        return f"""
        Number of images: {self.__number}\n
        World points: {self.world_points}\n
        Left image points: {self.left_ideal_points}\n
        Right image points: {self.right_ideal_points}\n
        Ideal parameters: {self.ideal_parameters}\n
        Offset parameters: {self.offset_parameters}\n
        Offset amount: {self.perturb_by}\n
        """
    
    def perturb_parameters(self, how_much):
        """
        Randomly perturb ideal case parameters by a set amount.
        """
        self.perturb_by = how_much
        for i, ideal_parameter in enumerate(self.ideal_parameters):
            self.offset_parameters[i] = ideal_parameter * np.random.uniform(1 - how_much, 1 + how_much)
    
    def copy_homography(self) -> Homography:
        """
        Return a shallow copy of the homography object.
        """
        return copy(self.homography)
    
    @property
    def number(self):
        return self.__number
    
    @number.setter
    def number(self, no_images):
        self.__number = no_images
    
    @property
    def homography(self):
        return self.__homography
