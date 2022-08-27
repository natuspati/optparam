# Imports
import numpy as np
import random
from scipy.optimize import least_squares, basinhopping, minimize, leastsq
from scipy.linalg import svd
import pprint
import pandas as pd

# Module settings
np.set_printoptions(precision=2, suppress=False)

# Constants
NO_INPUT_TERMS = 10
SCALING_ABNORMALITY1 = 1e0
SCALING_ABNORMALITY2 = 1e3
PERTURBATION_COEFFICIENT = 0.1
RANDOM_EXACT_SOLUTION = False
OPTIMIZATION_METHOD = 0  # 0 - LM, 1 - Basin hopping, 2 - BFGS


# Functions
def objective(parameters, input_list, output_list):
    n = len(input_list)
    error_vector = np.empty(n)
    
    for i in range(n):
        error_vector[i] = (forward(parameters, input_list[i]) - output_list[i])
    
    return error_vector


def scalar_objective(parameters, input_list, output_list):
    return np.linalg.norm(objective(parameters, input_list, output_list))


def objective_jac(parameters, input_list, output_list):
    return np.array(forward_jac(parameters, input_list, output_list))


def forward_jac(parameters, input_list, output_list):
    try:
        jac_matrix = []
        for input_var, output_var in zip(input_list, output_list):
            jac_matrix.append(jac2(*parameters, input_var, output_var))
    except TypeError:
        jac_matrix = jac2(*parameters, input_list)
    return jac_matrix


def forward(parameters, input_list):
    try:
        output_list = []
        for entry in input_list:
            output_list.append(func2(*parameters, entry))
    except TypeError:
        output_list = func2(*parameters, input_list)
    return output_list


def func(x1: float, x2: float, x3: float, input_var: float):
    return (np.cos(x1) * input_var + x2) * x3


def func1(x1: float, x2: float, input_var: float, const=1e-9):
    return (np.cos(x1) * input_var + x2) * const


def func2(x1: float, x2: float, x3: float, input_var: float):
    return (x1 * input_var + x2) * x3


def jac(x1: float, x2: float, x3: float, input_var: float, output_var: float):
    return [-2 * x3 * np.sin(x1) * (x2 * x3 + x3 * np.cos(x1) * input_var - output_var) * input_var,
            2 * x3 * (x2 * x3 + x3 * np.cos(x1) * input_var - output_var),
            2 * (x2 + np.cos(x1) * input_var) * (x2 * x3 + x3 * np.cos(x1) * input_var - output_var)]


def jac1(x1: float, x2: float, input_var: float, output_var: float, const=1e-9):
    return [-2 * const * np.sin(x1) * (const * x2 + const * np.cos(x1) * input_var - output_var) * input_var,
            2 * const * (const * x2 + const * np.cos(x1) * input_var - output_var)]


def jac2(x1: float, x2: float, x3: float, input_var: float, output_var: float):
    return [2 * x3 * (x2 * x3 + x1 * x3 * input_var - output_var) * input_var,
            2 * x3 * (x2 * x3 + x1 * x3 * input_var - output_var),
            2 * (x2 + x1 * input_var) * (x2 * x3 + x1 * x3 * input_var - output_var)]


def check_sensitivity(parameters, input_list, output_list):
    costs = np.empty(len(parameters))
    
    for i, parameter in enumerate(parameters):
        perturbed_parameters = parameters.copy()
        perturbed_parameters[i] += 1e-5
        costs[i] = scalar_objective(perturbed_parameters,
                                    input_list,
                                    output_list)
    
    return costs


def randomize(lst, percent=PERTURBATION_COEFFICIENT):
    new_lst = lst.copy()
    for i in range(len(lst)):
        new_lst[i] *= np.random.uniform(1 - percent, 1 + percent)
    return new_lst


def singular_values(dictionary):
    try:
        jac = dictionary.jac
        output_list = svd(jac)
        return output_list[1]
    except (AttributeError, ValueError):
        pass


def generate_data():
    # Generate input list of values.
    input_values = np.linspace(-2, 2, num=NO_INPUT_TERMS)
    
    # Generate exact solution, scale it and perturbed parameter list
    if RANDOM_EXACT_SOLUTION:
        exact_solution = np.array([np.pi, random.random(), random.random()])
    else:
        exact_solution = np.array([3.14, 1, 1])
    exact_solution *= np.array([1, SCALING_ABNORMALITY1, SCALING_ABNORMALITY2])
    
    initial_guess = randomize(exact_solution)
    
    # Generate putput list with random perturbation
    exact_output = forward(exact_solution, input_values)
    
    return input_values, exact_solution, initial_guess, exact_output


def optimize(exact_solution, initial_guess, input_values, exact_output, to_print=True):
    kwarg_dict = {"input_list": input_values,
                  "output_list": exact_output}
    
    # Calculate sensitivity of objective function to each parameter
    sensitivity = check_sensitivity(exact_solution, **kwarg_dict)
    
    # Minimize objective function and find exact solution
    if OPTIMIZATION_METHOD == 0:
        res = least_squares(objective,
                            initial_guess,
                            method="lm",
                            jac="2-point",
                            x_scale="jac",
                            verbose=1,
                            ftol=1e-12,
                            xtol=1e-12,
                            kwargs=kwarg_dict)
    elif OPTIMIZATION_METHOD == 1:
        minimizer_kwargs = {"method": "BFGS",
                            "jac": objective_jac,
                            "hess": "2-point",
                            "args": tuple(kwarg_dict.values())}
        res = basinhopping(scalar_objective,
                           initial_guess,
                           niter=10,
                           minimizer_kwargs=minimizer_kwargs)
    elif OPTIMIZATION_METHOD == 2:
        res = minimize(scalar_objective,
                       initial_guess,
                       args=tuple(kwarg_dict.values()),
                       options={"maxiter": 100, "disp": True})
    else:
        return None
    
    if to_print:
        print(f"""
            Initial guess: {initial_guess}
            Converged solution: {res.x},
            Exact solution: {exact_solution},
            Initial cost: {scalar_objective(initial_guess, **kwarg_dict):.1e},
            Final cost: {res.cost:.1e},
            Singular values of Jacobian: {' '.join(['{:.3e}'.format(x) for x in singular_values(res)])},
            Sensitivity to indices: {sensitivity},
            Perturbation in initial guess: {PERTURBATION_COEFFICIENT}.
            """)
    
    return res


def main():
    input_values, exact_solution, initial_guess, exact_output = generate_data()
    
    optimize(exact_solution, initial_guess, input_values, exact_output)


if __name__ == '__main__':
    # main()
    
    input_values, exact_solution, initial_guess, exact_output = generate_data()
    
    optimize(exact_solution, initial_guess, input_values, exact_output)
