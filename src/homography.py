import numpy as np
import transforms3d.reflections as tr
import transforms3d.affines as ta
from scipy.spatial.transform import Rotation as stR
from extra_data_types import ndarray


class Homography:
    def __init__(self,
                 intrinsic_matrix: ndarray,
                 inner_mirror_angle: float | int,
                 outer_mirror_angle: float | int,
                 dist_bw_mirrors: float | int,
                 dist_to_mirrors: float | int,
                 rotation_vectors: ndarray,
                 translation_vectors: ndarray):
        
        """
        Homography class keeps track of parameters within the optimization process and is responsible for
        reconstructing world points on image points. The class handles multi-image workflow.
        
        Parameters
        ----------
        intrinsic_matrix
            (3x3) matrix for intrinsic matrix of a camera.
        inner_mirror_angle
            Angle of inner mirrors wrt to z-normal in xz plane.
        outer_mirror_angle
            Angle of outer mirrors wrt to z-normal in xz plane.
        dist_bw_mirrors
            Distance between the bases of inner and outer mirrors.
        dist_to_mirrors
            Distance between the bases of mirrors and camera.
        rotation_vectors
            (number of images, 3) rotation vectors between world and camera coordinate systems for each image.
        translation_vectors
            (number of images, 3) translation vectors between world and camera coordinate systems for each image.
        """
        self.parameters, self.reflections, self.rt_matrices, self.projections = [None] * 4
        
        self.num_images = len(rotation_vectors)
        
        self.intrinsic = intrinsic_matrix
        
        angles, distances = self.calculate_mirror_parameters(inner_mirror_angle, outer_mirror_angle,
                                                             dist_bw_mirrors, dist_to_mirrors)
        
        initial_parameters = np.hstack((angles, distances,
                                        self._unpack_intrinsic(intrinsic_matrix),
                                        self._rearrange_target_parameters(rotation_vectors, translation_vectors)))
        
        self.update(initial_parameters)
    
    def update(self, new_parameters: ndarray):
        """
        Update values of parameters with a new vector.
        
        Parameters
        ----------
        new_parameters
            (number of image x 6 + 16) Vector of new parameters of the size.
        """
        self.parameters = new_parameters
        
        self.reflections = self._create_reflection_matrices()
        
        self.rt_matrices = self._create_rt_matrices()
        
        self.projections = self._create_projection_matrices()
    
    def project_to_images(self, world_points: ndarray) -> tuple[ndarray, ndarray]:
        """
        Project a set of world points in non-homogeneous coordinates to image coordinate for each image. Projections are
        stored in two arrays for left and right paths.
        
        Parameters
        ----------
        world_points
            World points in the global frame in non-homogeneous coordinates.
        """
        num_points = len(world_points)
        
        left_points_list = np.empty((self.num_images, num_points, 2))
        right_points_list = np.empty((self.num_images, num_points, 2))
        
        for i in range(self.num_images):
            left_projection_matrix = self.projections[i, 0]
            right_projection_matrix = self.projections[i, 1]
            for j, world_point in enumerate(world_points):
                homogeneous_point = self._homogenize(world_point)
                left_points_list[i, j] = self._project_point(left_projection_matrix, homogeneous_point)
                right_points_list[i, j] = self._project_point(right_projection_matrix, homogeneous_point)
        
        return left_points_list, right_points_list
    
    @classmethod
    def calculate_mirror_parameters(cls, inner_mirror_angle: float | int,
                                    outer_mirror_angle: float | int,
                                    dist_bw_mirrors: float | int,
                                    dist_to_mirrors: float | int) -> tuple[ndarray, ndarray]:
        """
        Calculate mirror parameters in spherical coordinates (:math:`{\\theta,\\phi,r}`) from measurable parameters of
        a four-mirror system. Mirror angles is a vector of size 8 for theta and phi. Distances is a vector of size 4.
        
        Parameters
        ----------
        inner_mirror_angle: float | int
            Angle between inner mirror and z-normal on xz-plane.
        outer_mirror_angle: float | int
            Angle between outer mirror and z-normal on xz-plane.
        dist_bw_mirrors: float | int
            Distance between the bases of inner and outer mirrors.
        dist_to_mirrors: float | int
            Distance between the bases of mirrors and camera.
        """
        inner_angle = inner_mirror_angle * np.pi / 180
        outer_angle = outer_mirror_angle * np.pi / 180
        theta_inner = np.pi / 2 + inner_angle
        theta_outer = np.pi / 2 + outer_angle
        phi = np.pi
        
        mirror_angles = np.array([[theta_inner, phi],
                                  [theta_outer, phi],
                                  [theta_inner, 0],
                                  [theta_outer, 0]])
        
        mirror_points = np.array([[0, 0, dist_to_mirrors],
                                  [-dist_bw_mirrors, 0, dist_to_mirrors],
                                  [0, 0, dist_to_mirrors],
                                  [dist_bw_mirrors, 0, dist_to_mirrors]])
        
        distances = np.empty(4)
        for i, (angle, mirror_point) in enumerate(zip(mirror_angles, mirror_points)):
            mirror_normal = cls._find_mirror_normal(angle)
            distances[i] = cls._find_r(mirror_normal, mirror_point)
        
        return mirror_angles.flatten(), distances
    
    def _create_projection_matrices(self) -> ndarray:
        """
        Create projection matrices for each image. Projection matrices of size (number of images x 2 x 3 x 4).
        Indices 0 and 1 in axis=1 are for left and right paths respectively.
        """
        projection_matrices = np.empty((self.num_images, 2, 3, 4))
        projective_matrix = np.hstack((np.eye(3), np.zeros((3, 1))))
        
        q_left_inner, q_left_outer, q_right_inner, q_right_outer = \
            self.reflections[0], self.reflections[1], self.reflections[2], self.reflections[3]
        
        for i in range(self.num_images):
            projection_matrices[i, 0] = \
                self.intrinsic.dot(projective_matrix.dot(q_left_inner.dot(q_left_outer.dot(self.rt_matrices[i]))))
            projection_matrices[i, 1] = \
                self.intrinsic.dot(projective_matrix.dot(q_right_inner.dot(q_right_outer.dot(self.rt_matrices[i]))))
        
        return projection_matrices
    
    def _create_rt_matrices(self) -> ndarray:
        """
        Create rotation and translation matrices per each image in homogeneous coordinates. The matrices have size
        (number of images x 4 x 4)
        """
        rt_matrices = np.empty((self.num_images, 4, 4))
        rotations_translations = self.parameters[16:].reshape((self.num_images, 6))
        for i, rt in enumerate(rotations_translations):
            rotation_object = stR.from_rotvec(rt[:3])
            rotation_matrix = rotation_object.as_matrix()
            translation_vector = rt[3:]
            rt_matrices[i] = ta.compose(translation_vector, rotation_matrix, [1, 1, 1])
        return rt_matrices
    
    def _create_reflection_matrices(self) -> ndarray:
        """
        Create 4 reflection matrices in homogeneous coordinates. Reflection matrices are stored in array of size
        (4 x 4 x 4). In axis=0, 0 - left inner, 1 - left outer, 2 - right inner, 3 - right outer.
        """
        reflection_matrices = np.empty((4, 4, 4))
        mirror_points, mirror_normals = self._calculate_mirror_points_normals()
        for i, (mirror_point, mirror_normal) in enumerate(zip(mirror_points, mirror_normals)):
            reflection_matrices[i] = tr.rfnorm2aff(mirror_normal, mirror_point)
        return reflection_matrices
    
    def _calculate_mirror_points_normals(self) -> tuple[ndarray, ndarray]:
        """
        Calculate mirror points and normal vectors from the spherical mirror coordinates (:math:`{\\theta,\\phi,r}`).
        Mirror points and normals are stored as arrays of size (4 x 3).
        """
        angles = self.parameters[:8].reshape((4, 2))
        distances = self.parameters[8:12]
        mirror_points = np.empty((4, 3))
        mirror_normals = np.empty((4, 3))
        for i, (angle, distance) in enumerate(zip(angles, distances)):
            mirror_normals[i] = self._find_mirror_normal(angle)
            mirror_points[i] = mirror_normals[i].dot(distance)
        return mirror_points, mirror_normals
    
    def _rearrange_target_parameters(self, rotation_vectors: ndarray,
                                     translation_vectors: ndarray) -> ndarray:
        """
        Rearrange a stack of rotation vectors and translation vectors in the following order:
        
        rotation vector for image 1, translation vector for image 1, rotation vector for image 2, ...
        
        The returned vector has length 6 x number of images.
        
        Parameters
        ----------
        rotation_vectors
            (number of images x 3) vertically stacked rotation vectors.
        translation_vectors
            (number of images x 3) vertically stacked translation vectors.
        """
        rotations_translations = np.empty((self.num_images, 6))
        for i, (rotation_vector, translation_vector) in enumerate(zip(rotation_vectors, translation_vectors)):
            rotations_translations[i] = np.hstack((rotation_vector, translation_vector))
        return rotations_translations.flatten()
    
    @staticmethod
    def _find_r(normal: ndarray, point: ndarray) -> ndarray:
        """
        Find the shortest distance using a normal of a plane and a point.
        
        Parameters
        ----------
        normal
            Normal vector of a plane.
        point
            Non-homogeneous point to which distance is calculated.
        """
        return np.array(normal).dot(point)
    
    @staticmethod
    def _find_mirror_normal(angles: ndarray) -> ndarray:
        """
        Convert angles of a normal of a plane to a unit normal according to:
        
        https://en.wikipedia.org/wiki/Spherical_coordinate_system#Coordinate_system_conversions
        
        Parameters
        ----------
        angles
            Two-element array in order theta, phi.
        """
        return np.array([np.cos(angles[1]) * np.sin(angles[0]),
                         np.sin(angles[1]) * np.sin(angles[0]),
                         np.cos(angles[0])])
    
    @staticmethod
    def _unpack_intrinsic(matrix: ndarray) -> ndarray:
        """
        Unpack 4 parameters from an intrinsic matrix in order: (:math:`{f_x, f_y, c_x, c_y}`).
        
        Parameters
        ----------
        matrix
            (3x3) intrinsic matrix.
        """
        return np.array([matrix[0, 0], matrix[1, 1], matrix[0, 2], matrix[1, 2]])
    
    @classmethod
    def _project_point(cls, projection_matrix: ndarray, world_point: ndarray) -> ndarray:
        """
        Multiply a projection matrix with a point in homogeneous coordinates.
        
        Parameters
        ----------
        projection_matrix
            (3x4) projection matrix.
        
        world_point
            World point in homogeneous coordinates.
        """
        return cls._non_homogenize(projection_matrix.dot(world_point))
    
    @staticmethod
    def _homogenize(point: ndarray) -> ndarray:
        """
        Convert a point to homogeneous coordinates.
        
        Parameters
        ----------
        point
            Point in non-homogeneous coordinates.
        """
        return np.append(point, 1).copy()
    
    @staticmethod
    def _non_homogenize(point: ndarray) -> ndarray:
        """
        Convert a point to non-homogeneous coordinates.
        
        Parameters
        ----------
        point: array
            Point in homogeneous coordinates.
        """
        return np.array([point[0] / point[2], point[1] / point[2]])
