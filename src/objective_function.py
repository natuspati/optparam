import numpy as np
from homography import Homography
from extra_data_types import ndarray


def objective(parameters: ndarray,
              world_points: ndarray,
              left_images_list: list,
              right_images_list: list,
              homography_object: Homography,
              scaling=None,
              scalar=False) -> ndarray | bool:
    """
    Objective function to minimize a set of intrinsic (camera nad mirror related) and extrinsic parameters. Depending
    on scalar, returns either error vector or its norm.
    
    Parameters
    ----------
    parameters
        A vector of the 16 intrinsic and (number of images x 6) extrinsic parameters.
    world_points
        World points in the global frame in non-homogeneous coordinates.
    left_images_list
        Left-half list of image points.
    right_images_list
        Right-half list of image points.
    homography_object
        Object to project world points to image points and store current values of projection matrices and their
        components.
    scaling
        Vector of scaling numbers for parameters.
    scalar
        If True, return norm of the error vector, otherwise return its norm.
    """
    if not scaling:
        scaling = np.ones_like(parameters)
    parameters /= scaling
    homography_object.update(parameters)
    
    left_reconstructed_list, right_reconstructed_list = homography_object.project_to_images(world_points)
    error_vector = construct_error_vector(left_images_list, right_images_list,
                                          left_reconstructed_list, right_reconstructed_list)
    
    if scalar:
        return np.linalg.norm(error_vector)
    
    return error_vector


def construct_error_vector(left_images_list: list,
                           right_images_list: list,
                           left_reconstructed_list: ndarray,
                           right_reconstructed_list: ndarray) -> ndarray:
    """
    Find error between reconstructed and image points.
    
    Parameters
    ----------
    left_images_list
        Left-half list of image points.
    right_images_list
        Right-half list of image points.
    left_reconstructed_list
        Left-half array of reconstructed points.
    right_reconstructed_list
        Left-half array of reconstructed points.
    """
    num_images = len(left_images_list)
    num_points = len(left_images_list[0])
    errors = np.empty((num_images, num_points, 2))
    
    for i in range(num_images):
        left_image = left_images_list[i]
        right_image = right_images_list[i]
        left_reconstructed = left_reconstructed_list[i]
        right_reconstructed = right_reconstructed_list[i]
        for j in range(num_points):
            errors[i, j, 0] = np.linalg.norm(left_image[j] - left_reconstructed[j])
            errors[i, j, 1] = np.linalg.norm(right_image[j] - right_reconstructed[j])
    
    return errors.flatten()


if __name__ == '__main__':
    pass
