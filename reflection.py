import numpy as np
import matplotlib.pyplot as plt
from system import System, TargetContainer, Camera, Mirror, TargetSystem
import transforms3d.reflections as tr

if __name__ == "__main__":
    points = np.array([[-1, 0, 5, 1],[1, 0, 4, 1], [-2, 0, 4.5, 1]])
    angle = 52 * np.pi / 180
    theta = 3 * np.pi / 2 + angle
    phi = np.pi
    r = 1 * np.cos(theta)
    mirror = Mirror(theta, phi, r)
    matrix = tr.rfnorm2aff(mirror.orientation, mirror.origin)
    ref_points = points.copy()
    ref_points_own = np.empty((3, 3))
    for i, point in enumerate(points):
        ref_points[i] = matrix.dot(point)
        ref_points_own[i] = mirror.reflect_point(point[:-1])

    # Plotting
    plt.close("all")
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.grid()

    ax.scatter([mirror.origin[0], 0], [mirror.origin[2], 0], c=['C1', 'C0'])
    ax.annotate(r'$O_{mir}$', (mirror.origin[0], mirror.origin[2]))
    ax.annotate(r'$O$', (0, 0))

    mir_norm_x = [mirror.origin[0], mirror.origin[0] + mirror.orientation[0]]
    mir_norm_z = [mirror.origin[2], mirror.origin[2] + mirror.orientation[2]]
    ax.plot(mir_norm_x, mir_norm_z, c='C1')
    ax.plot([0, 1], [0, 0], c='C0')
    ax.plot([0, 0], [1, 0], c='C0')
    ax.plot([1, -4], [0, 5], '--', c='C1')

    for i, (point, ref_point, ref_point_own) in enumerate(zip(points, ref_points, ref_points_own)):
        ax.scatter(point[0], point[2], c='C3')
        ax.annotate(r'$p$' + f'{i}', (point[0], point[2]))
        # ax.scatter(ref_point[0], ref_point[2], c='C3')
        # ax.annotate(r'$p_{ref, hom}$' + f'{i}', (ref_point[0], ref_point[2]))
        ax.scatter(ref_point_own[0], ref_point_own[2], c='C3')
        ax.annotate(r'$p_{ref, nonhom}$' + f'{i}', (ref_point_own[0], ref_point_own[2]))

    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$z$')

    plt.show()

    # theta1 = 2 * np.pi - np.pi / 4
    # theta2 = 2 * np.pi - 48 * np.pi / 180
    # phi = np.pi
    # d = 1
    # d1 = d - d / np.tan(48 * np.pi / 180)
    # r1 = d * np.cos(theta1)
    # r2 = d1 * np.cos(48 * np.pi / 180)
    # mirror1 = Mirror(theta1, phi, r1)
    # mirror2 = Mirror(theta2, phi, r2)
    #
    # point = np.array([-2, 0, 5])
    # ref_point1 = mirror2.reflect_point(point)
    # ref_point2 = mirror1.reflect_point(ref_point1)
    #
    # # Plotting
    # plt.close("all")
    # fig, ax = plt.subplots()
    # ax.set_aspect('equal')
    # ax.grid()
    #
    # ax.scatter([mirror1.origin[0], 0], [mirror1.origin[2], 0], c=['C1', 'C0'])
    # ax.annotate(r'$O_{mir}$', (mirror1.origin[0], mirror1.origin[2]))
    # ax.annotate(r'$O$', (0, 0))
    #
    # ax.scatter(mirror2.origin[0], mirror2.origin[2], c='C2')
    # ax.annotate(r'$O_{mir}$', (mirror2.origin[0], mirror2.origin[2]))
    # ax.annotate(r'$O$', (0, 0))
    #
    # mir_norm_x = [mirror1.origin[0], mirror1.origin[0] + mirror1.orientation[0]]
    # mir_norm_z = [mirror1.origin[2], mirror1.origin[2] + mirror1.orientation[2]]
    # mir1_norm_x = [mirror2.origin[0], mirror2.origin[0] + mirror2.orientation[0]]
    # mir1_norm_z = [mirror2.origin[2], mirror2.origin[2] + mirror2.orientation[2]]
    # ax.plot(mir_norm_x, mir_norm_z, c='C1')
    # ax.plot(mir1_norm_x, mir1_norm_z, c='C2')
    # ax.plot([0, 1], [0, 0], c='C0')
    # ax.plot([0, 0], [1, 0], c='C0')
    #
    # ax.scatter(point[0], point[2], c='C3')
    # ax.annotate(r'$p$', (point[0], point[2]))
    # ax.scatter(ref_point1[0], ref_point1[2], c='C3')
    # ax.annotate(r'$p_{ref}$', (ref_point1[0], ref_point1[2]))
    # ax.scatter(ref_point2[0], ref_point2[2], c='C3')
    # ax.annotate(r'$p_{ref1}$', (ref_point2[0], ref_point2[2]))
    #
    # ax.set_xlabel(r'$x$')
    # ax.set_ylabel(r'$z$')
    #
    # plt.show()
    #
    # # fig = plt.figure()
    # # ax = fig.add_subplot(projection='3d')
    # # ax.set_box_aspect(aspect = (1,1,1))
    # # px, py, pz = mirror.origin
    # # xline = [px, px + mirror.orientation[0]]
    # # yline = [py, py + mirror.orientation[1]]
    # # zline = [pz, pz + mirror.orientation[2]]
    # # ax.scatter(px, py, pz)
    # # ax.scatter(0, 0, 0, color='C1')
    # # ax.plot3D(xline, yline, zline)
    # # ax.plot3D([0, 1], [0, 0], [0, 0], color='C1')
    # # ax.plot3D([0, 0], [0, 1], [0, 0], color='C1')
    # # ax.plot3D([0, 0], [0, 0], [0, 1], color='C1')
