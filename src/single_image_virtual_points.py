# Imports
import numpy as np
from data_generator import DataGenerator
from optimizer import Optimizer
from initialization_constants import INITIALIZATION_CONSTANTS

# Module settings
np.set_printoptions(precision=2, suppress=True, threshold=3)

# Constants
NO_IMAGES = 1
PERTURB_BY = 0.01
lsq_dict = {"method": "trf",
            "jac": "3-point",
            "x_scale": "jac",
            "verbose": 0,
            "ftol": 1e-12,
            "xtol": 1e-12,
            "max_nfev": 1000}


# Functions
def main():
    # Generate data.
    data = DataGenerator(NO_IMAGES, PERTURB_BY, *INITIALIZATION_CONSTANTS)
    
    # Call optimizer and store data in a wrapped child class of Scipy.OptimizeResult.
    optimizer = Optimizer(lsq_dict, data)
    result = optimizer.least_squares()
    
    return result


if __name__ == '__main__':
    res = main()
    print(res)
