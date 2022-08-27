# Imports
import numpy as np
import random
from copy import copy
from src.target_types import Checkerboard
from scipy.spatial.transform import Rotation
from scipy.optimize import basinhopping
from classes import Homography


def obj_fun(parameters, scaling, left_image_list, right_image_list, world_points, hom_object):
    parameters /= scaling
    hom_object.update(parameters)
    
    num_images = len(left_image_list)
    num_points = len(world_points)
    
    error_vector = np.zeros(2 * num_points * num_images)
    
    rot_tran_vectors = parameters[16:]
    rot_tran_vectors = rot_tran_vectors.reshape(len(rot_tran_vectors) // 3, 3)
    rot_vecs = rot_tran_vectors[::2]
    tran_vecs = rot_tran_vectors[1::2]
    left_reconstructed, right_reconstructed = hom_object.project_to_image(world_points, rot_vecs, tran_vecs)
    
    for i in range(num_images):
        left_img = left_image_list[i]
        right_img = right_image_list[i]
        left_rec = left_reconstructed[i]
        right_rec = right_reconstructed[i]
        for j in range(num_points):
            k_left = 2 * (j + num_points * i)
            k_right = k_left + 1
            error_vector[k_left] = np.linalg.norm(left_img[j] - left_rec[j])
            error_vector[k_right] = np.linalg.norm(right_img[j] - right_rec[j])
    for i in range(len(error_vector)):
        if i % 140 == 0:
            error_vector = np.delete(error_vector, [i, i + 1], 0)
    return error_vector


def wrapped_obj_fun(parameters, scaling, left_image_list, right_image_list, world_points, hom_object):
    return np.linalg.norm(obj_fun(parameters, scaling, left_image_list, right_image_list, world_points, hom_object))


def calculate_scaling(ideal_parameters, err_fun, parameter_dict):
    """
    Calculate proper scaling vector for parameters.
    """
    costs = np.empty((len(ideal_parameters)))
    for i in range(len(ideal_parameters)):
        perturbed_parameters = ideal_parameters.copy()
        perturbed_parameters[i] += 1E-5
        parameter_dict["parameters"] = perturbed_parameters
        costs[i] = err_fun(obj_fun, parameter_dict)
    return costs


if __name__ == "__main__":
    """
    Initialize ideal world coordinate points from already written code in class Targets.
    Grid points are contained in 'gridpoints' attribute.
    """
    target_pattern = Checkerboard(8, 11, 15)
    
    """
    Generate ideal case of rotation and translation transformation matrices.
    Assume 4 images.
    """
    euler_angles = np.array([[90, 0, 0]], dtype=float)
    
    translation_vectors = np.array([[0, 0, 300]], dtype=float)
    
    # Convert euler angles to rotational vector to be used as a parameter.
    rotational_vectors = euler_angles.copy()
    rotational_matrices = []
    for i, euler_angle in enumerate(euler_angles):
        rotations_object = Rotation.from_euler('zyx', euler_angle, degrees=True)
        rotational_vectors[i] = rotations_object.as_rotvec()
        rotational_matrices.append(rotations_object.as_matrix())
    
    """
    Generate intrinsic parameters
    """
    fx, fy, cx, cy = 1000.0, 1000.0, 400.0, 300.0
    intrinsic_matrix = np.array([[fx, 0, cx, 0],
                                 [0, fy, cy, 0],
                                 [0, 0, 1, 0]])
    
    """
    Generate mirror parameters from symmetric case.
    Inner mirror = 45deg, outer mirror = 52deg,
    distance between mirrors = 20, distance between mirrors and camera = 20.
    """
    inner_angle = 45 * np.pi / 180
    outer_angle = 52 * np.pi / 180
    theta_inner = np.pi / 2 + inner_angle
    theta_outer = np.pi / 2 + outer_angle
    phi = np.pi
    mirror_angles = np.array([[theta_inner, phi],
                              [theta_outer, phi],
                              [theta_inner, 0],
                              [theta_outer, 0]])
    
    dist_bw_mirrors = 20
    dist_to_mirrors = 20
    mirror_points = np.array([[0, 0, dist_to_mirrors],
                              [-dist_bw_mirrors, 0, dist_to_mirrors],
                              [0, 0, dist_to_mirrors],
                              [dist_bw_mirrors, 0, dist_to_mirrors]])
    
    """
    To find distance parameters, invoke Homography object and get parameters.
    Set rotations and translations to the first case.
    """
    throwaway_homography = Homography(rotational_matrices[0],
                                      translation_vectors[0],
                                      mirror_angles,
                                      mirror_points,
                                      intrinsic_matrix)
    
    mirror_related_parameters = throwaway_homography.get_parameters()[:12]
    
    """
    Construct ideal case parameter vector as follows:
        8 mirror angle parameters
        4 mirror distance parameters
        3 rotations for case 1
        3 translations for case 1
        3 rotations for case 2
        3 translations for case 2
        ...
    """
    ideal_case_parameters = np.hstack((mirror_related_parameters, [fx, fy, cx, cy]))
    for i, (rotational_vector, translation_vector) in enumerate(zip(rotational_vectors, translation_vectors)):
        ideal_case_parameters = np.append(ideal_case_parameters, [rotational_vector, translation_vector])
    
    """
    Generate ideal image points with ideal case parameters.
    """
    ideal_left_image_points, ideal_right_image_points = throwaway_homography.project_to_image(target_pattern.gridpoints,
                                                                                              rotational_vectors,
                                                                                              translation_vectors)
    
    """
    Perturb the ideal case solution by 1% and see difference between converged and ideal case solutions.
    """
    percent = 0.01
    perturbed_parameters = ideal_case_parameters.copy()
    for i, perturbed_parameter in enumerate(perturbed_parameters):
        if np.isclose(perturbed_parameter, 0):
            pass
        else:
            perturbed_parameters[i] *= random.uniform(1 - percent, 1 + percent)
    
    parameter_dict = {"parameters": ideal_case_parameters,
                      "scaling": np.ones(22),
                      "left_image_list": ideal_left_image_points,
                      "right_image_list": ideal_right_image_points,
                      "world_points": target_pattern.gridpoints,
                      "hom_object": copy(throwaway_homography)}
    
    coefs = calculate_scaling(ideal_case_parameters, wrapped_obj_fun, parameter_dict)
    
    # new_ideal_parameters =  ideal_case_parameters * coefs
    # parameter_dict["scaling"] = coefs
    
    """
    Perturb the ideal case solution by 1% and see difference between converged and ideal case solutions.
    """
    percent = 0.01
    new_perturbed_parameters = ideal_case_parameters.copy()
    for i, new_perturbed_parameter in enumerate(new_perturbed_parameters):
        if np.isclose(new_perturbed_parameter, 0):
            pass
        else:
            new_perturbed_parameters[i] *= random.uniform(1 - percent, 1 + percent)
    parameter_dict["parameters"] = new_perturbed_parameters
    
    """
    Minimization callout.
    """
    minimizer_kwargs = {"method": "trust-constr",
                        "args": (coefs,
                                 ideal_left_image_points,
                                 ideal_right_image_points,
                                 target_pattern.gridpoints,
                                 copy(throwaway_homography)),
                        "jac": "3-point",
                        "options": {"maxiter": 3, "disp": True}}
    
    opt_res = basinhopping(wrapped_obj_fun, perturbed_parameters, minimizer_kwargs=minimizer_kwargs, niter=2)
    # opt_res = least_squares(obj_fun, ideal_case_parameters, method='lm', max_nfev=100, verbose=2,
    #                         args=(coefs,
    #                               ideal_left_image_points,
    #                               ideal_right_image_points,
    #                               target_pattern.gridpoints,
    #                               copy(throwaway_homography)))
    print(opt_res.x)
