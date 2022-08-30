# Imports
import numpy as np
from scipy.optimize import least_squares, minimize

# Constants
no_points = 10
no_parameters = 2
lsq_dict = {"method": "trf",
            "jac": "2-point",
            "verbose": 0,
            "ftol": 1e-12,
            "xtol": 1e-12,
            "max_nfev": 100}


# Functions
def objective(x, a):
    top = np.linalg.norm(a.dot(x)) ** 2
    # bot = np.linalg.norm(x)**2
    bot = 1
    return top / bot


def con(x):
    return np.linalg.norm(x)**2 - 1


def generate_data():
    # x = np.array([random_sign(i) for i in range(1, no_parameters + 1)])
    a = np.random.randint(-10, 10, size=(no_points, no_parameters))
    x0 = np.ones(no_parameters, dtype=float)
    return a, x0


def random_sign(number=1.0):
    return float((2 * np.random.randint(0, 2) - 1) * number)


def svd(a):
    u, s, vh = np.linalg.svd(a)
    v = vh.transpose()
    return v[:, -1]


def main():
    a, x0 = generate_data()
    # lsq_dict["args"] = [a]
    # result = least_squares(objective, x0, **lsq_dict)
    # result = minimize(objective, x0, args=(a), jac="2-point")
    cons = {"type": "eq",
            "fun": con}
    result = minimize(objective, x0, args=(a), method="SLSQP", jac="3-point", constraints=cons)
    return result.x, a, x0


if __name__ == '__main__':
    x, a, x0 = main()
    
    v = svd(a)
    
    vt = svd(a.transpose().dot(a))
    
    print(f"Solution from iterative: {x}\nSolution from SVD: {v}\nRatio from iterative: {x[0] / x[1]}\n"
          f"Ratio from SVD: {v[0] / v[1]}")
