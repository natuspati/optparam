# Imports
import numpy as np
from copy import copy
from classes import Homography, SingleCameraCalibrator
from scipy.spatial.transform import Rotation as R
from src.target_types import Checkerboard
from classes import ImageContainer


def obj_fun(parameters, img_points_left, img_points_right, obj_points, hom_mtx):
    hom_mtx.update(parameters)

    num_points = len(obj_points)
    reconstructed_left_points = img_points_left.copy()
    reconstructed_right_points = img_points_right.copy()

    error_vector = np.zeros(2 * num_points)

    for index, (obj_point, point_left, point_right) in enumerate(zip(obj_points, img_points_left, img_points_right)):
        i = 2 * index
        j = i + 1

        obj_point = np.hstack((obj_point, 1))
        reconstructed_left = non_homogeneous(hom_mtx.left_projection_matrix.dot(obj_point))
        reconstructed_right = non_homogeneous(hom_mtx.right_projection_matrix.dot(obj_point))

        reconstructed_left_points[index] = reconstructed_left
        reconstructed_right_points[index] = reconstructed_right

        error_vector[i] = np.linalg.norm(point_left - reconstructed_left)
        error_vector[j] = np.linalg.norm(point_right - reconstructed_right)

    return error_vector


def obj_fun1(parameters, img_points_left, img_points_right, obj_points, hom_mtx):
    tempvars = parameters.copy() * scal
    hom_mtx.update(tempvars)

    num_points = len(obj_points)
    reconstructed_left_points = img_points_left.copy()
    reconstructed_right_points = img_points_right.copy()

    error_vector = np.zeros(2 * num_points)

    for index, (obj_point, point_left, point_right) in enumerate(zip(obj_points, img_points_left, img_points_right)):
        i = 2 * index
        j = i + 1

        obj_point = np.hstack((obj_point, 1))
        reconstructed_left = non_homogeneous(hom_mtx.left_projection_matrix.dot(obj_point))
        reconstructed_right = non_homogeneous(hom_mtx.right_projection_matrix.dot(obj_point))

        reconstructed_left_points[index] = reconstructed_left
        reconstructed_right_points[index] = reconstructed_right

        error_vector[i] = np.linalg.norm(point_left - reconstructed_left)
        error_vector[j] = np.linalg.norm(point_right - reconstructed_right)

    return np.linalg.norm(error_vector)


def backward_pass(obj_point_vars, parameters, left_points, right_points, hom_mtx):
    hom_mtx.update(parameters)

    obj_point_vars = obj_point_vars.reshape((70, 3))

    num_points = len(obj_point_vars)
    reconstructed_left_points = left_points.copy()
    reconstructed_right_points = right_points.copy()
    error_vector = np.zeros(2 * num_points)

    for index, (obj_point, point_left, point_right) in enumerate(zip(obj_point_vars, left_points, right_points)):
        i = 2 * index
        j = i + 1

        obj_point = np.hstack((obj_point, 1))
        reconstructed_left = non_homogeneous(hom_mtx.left_projection_matrix.dot(obj_point))
        reconstructed_right = non_homogeneous(hom_mtx.right_projection_matrix.dot(obj_point))

        reconstructed_left_points[index] = reconstructed_left
        reconstructed_right_points[index] = reconstructed_right

        error_vector[i] = np.linalg.norm(point_left - reconstructed_left)
        error_vector[j] = np.linalg.norm(point_right - reconstructed_right)

    return error_vector


def forward_pass(params, obj_points, hom_mtx):
    hom_mtx.update(params)

    num_points = len(obj_points)
    reconstructed_left_points = np.empty((num_points, 2))
    reconstructed_right_points = reconstructed_left_points.copy()

    for i, obj_point in enumerate(obj_points):
        obj_point = np.hstack((obj_point, 1))
        reconstructed_left = non_homogeneous(hom_mtx.left_projection_matrix.dot(obj_point))
        reconstructed_right = non_homogeneous(hom_mtx.right_projection_matrix.dot(obj_point))
        reconstructed_left_points[i] = reconstructed_left
        reconstructed_right_points[i] = reconstructed_right

    return reconstructed_left_points, reconstructed_right_points


def non_homogeneous(point):
    return np.array([point[0]/point[2], point[1]/point[2]])


if __name__ == "__main__":
    scal = np.ones(22)
    scal[:8] = np.full(8, 1E-3)
    # scal = 1 / np.array([4.96807487e-05, 8.33563582e-05, 5.04922857e-05, 1.05724151e-04,
    #    5.09898288e-05, 8.68760288e-05, 5.18018968e-05, 1.11232797e-04,
    #    6.17612754e-02, 5.10607902e-02, 5.83279594e-02, 4.91260500e-02,
    #    3.01141394e-01, 9.57860124e-01, 8.45154251e-02, 8.45154258e-02,
    #    4.53844258e-03, 1.12135353e-02, 4.51492600e-04, 4.26901642e-02,
    #    4.40645629e-02, 3.74593394e-01])

    # Initialize the target
    target_pattern = Checkerboard(8, 11, 15)

    # Perform single camera calibration to obtain intrinsic matrix
    mono_calib = SingleCameraCalibrator("twodimgs1", "*.JPG", target_pattern)

    # Initialize homography object
    r = R.from_euler('zyx', [90, 0, 0], degrees=True)
    # r = R.from_rotvec(np.average(mono_calib.rvecs, 0).reshape(3))
    rot_mtx = r.as_matrix()
    # tran_vec = np.array([0, 0, 10])
    tran_vec = np.average(mono_calib.tvecs, 0).reshape(3)

    inner_angle = 45 * np.pi / 180
    outer_angle = 52 * np.pi / 180
    theta_inner = np.pi / 2 + inner_angle
    theta_outer = np.pi / 2 + outer_angle
    phi = np.pi
    ang = np.array([[theta_inner, phi],
                    [theta_outer, phi],
                    [theta_inner, 0],
                    [theta_outer, 0]])

    dist_bw_mirrors = 20
    dist_to_mirrors = 20
    pts = np.array([[0, 0, dist_to_mirrors],
                    [-dist_bw_mirrors, 0, dist_to_mirrors],
                    [0, 0, dist_to_mirrors],
                    [dist_bw_mirrors, 0, dist_to_mirrors]])

    P = Homography(rot_mtx, tran_vec, ang, pts, mono_calib.homogeneous_intrinsics())
    init_guess = P.get_parameters()
    conv_params = init_guess.copy()

    img_path = "testimgs1/good/good1"
    img_ext = "*.JPG"
    img_con = ImageContainer(img_path, img_ext)
    img_size = img_con.imgsize
    img_con.extract(target_pattern)
    num_points = len(img_con.objpoints)

    # opt_res = obj_fun(init_guess,
    #                   np.array(img_con.imgpoints_right).reshape((num_points, 2)),
    #                   np.array(img_con.imgpoints_left).reshape((num_points, 2)),
    #                   np.array(img_con.objpoints),
    #                   P)

    img_left_points = np.array(img_con.imgpoints_right).reshape((num_points, 2))
    img_right_points = np.array(img_con.imgpoints_left).reshape((num_points, 2))
    obj_points = np.array(img_con.objpoints)

    idealized_left_points, idealized_right_points = forward_pass(conv_params, obj_points, copy(P))

    # coefs = []
    # for i in range(len(init_guess)):
    #     preturbed_params = conv_params.copy() / scal
    #     preturbed_params[i] = preturbed_params[i] + 1E-5
    #     coefs.append(np.linalg.norm(obj_fun(preturbed_params, idealized_left_points, idealized_right_points, obj_points, P)))
    #     print(f"{np.linalg.norm(obj_fun1(preturbed_params, idealized_left_points, idealized_right_points, obj_points, copy(P))):.3E}")
    #
    # coefs = np.array(coefs)
    # scal = 1E-5/coefs
    #
    # conv_params = conv_params / scal

    preturbed_params = conv_params.copy() / scal

    # for i in range(len(init_guess)):
    #         preturbed_params[i] = preturbed_params[i] * 1.0001
    #
    # minimizer_kwargs = {"method": "trust-constr",
    #                     "args": (idealized_left_points, idealized_right_points, obj_points, copy(P)),
    #                     "jac": "3-point",
    #                     "options": {"maxiter": 30, "disp": True}}
    #
    # for iternum in [5]:
    #     # opt_res = minimize(obj_fun1, conv_params, method='trust-constr', jac='3-point', options={"maxiter": iternum, "disp": True},
    #     #                 args=(idealized_left_points, idealized_right_points, obj_points, copy(P)))
    #     opt_res = basinhopping(obj_fun1, preturbed_params, minimizer_kwargs=minimizer_kwargs, niter=iternum)
    #
    # # print(np.linalg.norm(obj_fun(conv_params, idealized_left_points, idealized_right_points, obj_points, copy(P))))
    # # for iternum in [200]:
    # #     minimizer_kwargs = {"method": "BFGS"}
    # #
    # #     # opt_res = basinhopping(obj_fun1, init_guess1, minimizer_kwargs=minimizer_kwargs,
    #     #                    niter=iternum)
    # #
    # #     opt_res = least_squares(obj_fun, preturbed_params, method='lm', max_nfev=iternum,
    # #                             verbose=0,
    # #                             args=(idealized_left_points, idealized_right_points, obj_points, copy(P)))
    # #
    #
    #     newsol = opt_res.x.copy()
    #     reldif = init_guess.copy()
    #     for i in range(len(init_guess)):
    #         if np.isclose(conv_params[i], 0):
    #             reldif[i] = abs(conv_params[i] - newsol[i] * scal[i])
    #         else:
    #             reldif[i] = abs(conv_params[i] - newsol[i] * scal[i]) / abs(conv_params[i])
    #
    #     reldif1 = init_guess.copy()
    #     for i in range(len(init_guess)):
    #         if np.isclose(conv_params[i], 0):
    #             reldif1[i] = abs(conv_params[i] - preturbed_params[i] * scal[i])
    #         else:
    #             reldif1[i] = abs(conv_params[i] - preturbed_params[i] * scal[i]) / abs(conv_params[i])
    #
    #     print(f"Number of allowed iters: {iternum*100}")
    #     print(f"Rel dif in the initial guess: {np.linalg.norm(reldif1):.2E}")
    #     print(f"Rel dif in the converged solution: {np.linalg.norm(reldif):.2E}")
    #
    #     print(f"Norm of error with preturbed solution: {np.linalg.norm(obj_fun1(preturbed_params, idealized_left_points, idealized_right_points, obj_points, copy(P))):.2E}")
    #     print(f"Norm of error with converged solution: {np.linalg.norm(obj_fun1(newsol, idealized_left_points, idealized_right_points, obj_points, copy(P))):.2E} \n")


    # plt.close("all")
    # fig, ax = plt.subplots()
    # ax.scatter(idealized_left_points[:,0], idealized_left_points[:, 1], c='C0')
    # ax.scatter(idealized_right_points[:, 0], idealized_right_points[:, 1], c='C1')
    # plt.show()

    # for i in range(len(obj_points)):
    #     idealized_left_points[i] = idealized_left_points[i] + np.array([1, 0])
    #     # idealized_right_points[i] = idealized_right_points[i] + np.array([-1, 0])


    # opt_res = least_squares(backward_pass, obj_points.flatten(), method='trf', max_nfev=100,
    #                         x_scale='jac', verbose=2,
    #                         args=(init_guess, img_left_points, img_right_points, P))
    #
    # opt_res1 = least_squares(obj_fun, init_guess, method='trf', max_nfev=100,
    #                         x_scale='jac', verbose=2,
    #                         args=(img_left_points, img_right_points, opt_res.x.reshape((70, 3)), P))

    # opt_res = least_squares(obj_fun, init_guess, method='trf', max_nfev=100,
    #                         x_scale='jac', verbose=2,
    #                         args=(img_left_points, img_right_points, obj_points, P))
    #
    # u, s, v = np.linalg.svd(opt_res.jac)
    # print(s)
    #
    # # tzs = np.linspace(300, 800, 50)
    # # costs = []
    # # for tz in tzs:
    # #     init_guess[-1] = tz
    # #     opt_res = least_squares(obj_fun, init_guess, method='trf', max_nfev=100,
    # #                             x_scale='jac',
    # #                             args=(img_left_points, img_right_points, obj_points, P))
    # #     costs.append(opt_res.cost)
    #
    # # jac = opt_res.jac
    # # U, s, Vh = np.linalg.svd(jac)
    #
    # # Plotting
    # plt.close("all")
    # fig, ax = plt.subplots()
    #
    # ax.semilogy(s)
    # ax.grid()
    # ax.set_xlabel("Singular values index")
    # ax.set_ylabel("Values")
    # # plt.savefig("reflection plots/semilog_singular_values.png")
    # plt.show()
    #
    # # ax.plot(tzs, costs)
    # # ax.grid()
    # # ax.set_xlabel(r"Initial guess for $T_z$")
    # # ax.set_ylabel("Cost after 100 iterations")
    # # plt.savefig("reflection plots/cost_vs_tz.png")
    #
    # # img_con = ImageContainer("testimgs1/good", img_ext)
    # # for i, impath in enumerate(img_con.stereoimgs):
    # #     img = cv.imread(impath)
    # #     [img_left, img_right] = img_con._img_split(img)
    # #     plt.close("all")
    # #     plt.imshow(img_left)
    # #     plt.savefig(f"left{i}.png")
    # #     plt.close("all")
    # #     plt.imshow(img_right)
    # #     plt.savefig(f"right{i}.png")

    # plt.close("all")
    # fig, ax = plt.subplots()
    # ax.scatter(obj_points[:,0], obj_points[:,1], c='C1')
    # ax.grid(which="both")
    # ax.set_xlabel("X")
    # ax.set_ylabel("Y")