# Imports
from scipy.optimize import least_squares
from data_generator import DataGenerator
from objective_function import objective
from initialization_constants import INITIALIZATION_CONSTANTS

# Constants
NO_IMAGES = 1
lsq_dict = {"method": "lm",
            "jac": "2-point",
            "x_scale": "jac",
            "verbose": 1,
            "ftol": 1e-12,
            "xtol": 1e-12,
            "max_nfev": 10}


# Functions
def main():
    # Generate data
    data = DataGenerator(NO_IMAGES, *INITIALIZATION_CONSTANTS)
    homography = data.copy_homography()
    
    # Minimization callout.
    lsq_dict["kwargs"] = {"world_points": data.world_points,
                          "left_images_list": data.left_ideal_points,
                          "right_images_list": data.right_ideal_points,
                          "homography_object": homography}
    least_squares(objective, data.offset_parameters, **lsq_dict)


if __name__ == '__main__':
    main()
