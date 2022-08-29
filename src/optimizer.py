from scipy.optimize import least_squares, OptimizeResult
from data_generator import DataGenerator
from objective_function import objective


class Optimizer:
    def __init__(self, options_dict: dict, data: DataGenerator):
        """
        Optimization callout. Uses Scipy's least squares at the moment with plans to expand to basin hopping algorithm.
        
        Parameters
        ----------
        options_dict
            Dictionary of options to control optimization parameters and arguments of the objective function.
        data
            Object that contains synthetic data for world and image points, exact and perturbed solutions.
        """
        self.initial_guess = data.offset_parameters
        self.exact_solution = data.ideal_parameters
        
        self.homography = data.copy_homography()
        
        self.options_dict = options_dict
        self.options_dict["kwargs"] = {"world_points": data.world_points,
                                       "left_images_list": data.left_ideal_points,
                                       "right_images_list": data.right_ideal_points,
                                       "homography_object": self.homography}
    
    def least_squares(self):
        """
        Call least squares optimization and store result in a wrapper of Scipy.OptimizationResult.
        """
        return Result(self, least_squares(objective, self.initial_guess, **self.options_dict))


class Result:
    def __init__(self, optimizer: Optimizer, result: OptimizeResult):
        """
        Result object which contains essential information from data, optimization settings and its results.
        
        Parameters
        ----------
        optimizer
            Optimizer object which initialized the least squares algorithm.
        result
            Scipy.OptimizationResult produced by least squares call.
        """
        self.__dict__.update(optimizer.__dict__)
        self._result = result
    
    def __getattr__(self, attr):
        """
        Get attributes contained within protected Scipy.OptimizationResult attribute.
        """
        if attr in self.__dict__:
            return getattr(self, attr)
        else:
            return getattr(self._result, attr)
    
    def __str__(self):
        """
        Print the essential data of the optimization result.
        """
        return (f"""
        Chosen optimization scheme: {self.options_dict["method"]}
        Initial guess: {self.initial_guess}
        Converged parameters: {self.x}
        Exact solution: {self.exact_solution}
        Final cost: {self.cost},
        Number of iterations: {self.njev},
        Exit flag: {self.message}
        """)


if __name__ == '__main__':
    pass
