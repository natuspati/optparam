# Imports
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from scipy.linalg import rq

# Module settings
np.set_printoptions(precision=2, suppress=True)

# Constants
RANDOMIZE_INPUTS = True
RANDOMIZE_COEFFICIENT = 0.1


# Functions
def main():
    # Construct a projection matrix
    intrinsic_matrix = np.array([[100, 0, 50],
                                 [0, 100, 50],
                                 [0, 0, 1]])
    
    rotation_vector = np.array([0, 0, np.pi / 2])
    
    translation_vector = np.array([100, 100, 100])
    
    if RANDOMIZE_INPUTS:
        # noinspection PyTypeChecker
        randomize([intrinsic_matrix, rotation_vector, translation_vector])
    
    homogeneous_intrinsic = homogenize(intrinsic_matrix)
    projection_matrix = construct_projection_matrix(homogeneous_intrinsic, rotation_vector, translation_vector)
    projection_matrix[abs(projection_matrix) < 1e-12] = 0
    
    # Perform decomposition as per slides
    kr_matrix = projection_matrix[:, :-1]
    kt_vector = projection_matrix[:, -1]
    
    r, q = unique_rq(kr_matrix)
    rotation_matrix_found = Rot.from_matrix(q)
    rotation_vector_found = rotation_matrix_found.as_rotvec()
    
    t_found = np.linalg.inv(r).dot(kt_vector)
    
    print(f"Found translation ,0vector: {t_found},\ninitial translation vector: {translation_vector}\n"
          f"Found rotation vector: {rotation_vector_found},\ninitial rotation vector: {rotation_vector}\n"
          f"Found intrinsic matrix:\n{r}\ninitial intrinsic matrix:\n{intrinsic_matrix}")


def construct_projection_matrix(intrinsic_matrix, rotation_vector, translation_vector):
    rt_matrix = construct_rt_matrix(rotation_vector, translation_vector)
    return intrinsic_matrix.dot(rt_matrix)


def construct_rt_matrix(rotation_vector, translation_vector):
    r = Rot.from_rotvec(rotation_vector)
    rotation_matrix = r.as_matrix()
    
    rt_matrix = np.vstack((np.hstack((rotation_matrix,
                                      translation_vector[:, np.newaxis])),
                           np.hstack((np.zeros(3), 1))))
    
    return rt_matrix


def unique_rq(matrix):
    r, q = rq(matrix)
    signs = 2 * (np.diag(r) >= 0) - 1
    r *= signs[np.newaxis, :]
    q *= signs[:, np.newaxis]
    return r, q


def homogenize(matrix):
    return np.hstack((matrix, np.zeros((len(matrix), 1))))


def randomize(input_var):
    try:
        for i in range(len(input_var)):
            input_var[i] = randomize(input_var[i])
        
        if hasattr(input_var, "ndim") and input_var.ndim == 2:
            input_var[2, 2] = 1
    
    except TypeError:
        input_var *= np.random.uniform(1 - RANDOMIZE_COEFFICIENT, 1 + RANDOMIZE_COEFFICIENT)
    
    return input_var


if __name__ == '__main__':
    main()
