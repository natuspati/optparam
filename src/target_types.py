from dataclasses import dataclass
import numpy as np


@dataclass
class TargetType:
    def __init__(self, verticals: int, horizontals: int, distance: float | int):
        """
        Sets up shape of the grid points on a target.
        
        Parameters
        ----------
        verticals
            Number of vertical points.
        horizontals
            Number of horizontal points.
        distance
            The closest distance between two points on the grid in mm.
        """
        self.verticals = verticals
        self.horizontals = horizontals
        self.distance = distance


class Checkerboard(TargetType):
    def __init__(self, verticals, horizontals, distance):
        """
        Checkerboard grid. The points only include inside edges.
        
        Parameters
        ----------
        verticals
            Number of vertical points.
        horizontals
            Number of horizontal points.
        distance
            The closest distance between two points on the grid in mm.
        """
        super().__init__(verticals, horizontals, distance)
        self.verticals = self.verticals - 1
        self.horizontals = self.horizontals - 1
        self.points = np.zeros((self.verticals * self.horizontals, 3), np.float64)
        self.points[:, :2] = self.distance * np.mgrid[0:self.verticals, 0:self.horizontals].T.reshape(-1, 2)


class Circles(TargetType):
    def __init__(self, verticals, horizontals, distance):
        """
        Asymmetric circles grid. The points only include inside edges.
        
        Parameters
        ----------
        verticals
            Number of vertical points.
        horizontals
            Number of horizontal points.
        distance
            The closest distance between two points on the grid in mm.
        """
        super().__init__(verticals, horizontals, distance)
        
        self.pattern_size = (self.verticals, self.horizontals)
        self.distance *= np.sqrt(2)
        num_points = int(self.verticals * (self.horizontals - 1) / 2 + self.verticals * (self.horizontals + 1) / 2)
        
        self.points = np.zeros((num_points, 3))
        counter = 0
        for i in range(self.horizontals):
            if i % 2 == 0:
                for j in range(self.verticals):
                    point = np.array([2 * j, i, 0])
                    self.points[counter] = point
                    counter += 1
            else:
                for j in range(self.verticals):
                    point = np.array([2 * j + 1, i, 0])
                    self.points[counter] = point
                    counter += 1
        
        self.points *= self.distance


if __name__ == "__main__":
    pass
