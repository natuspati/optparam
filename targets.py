# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 09:54:22 2021

@author: bekdulnm
"""

from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt


@dataclass
class Target(object):
    def __init__(self, verticals, horizontals, distance):
        self.verticals = verticals
        self.horizontals = horizontals
        self.distance = distance
        self.gridpoints = []


class Checkerboard(Target):
    def __init__(self, verticals, horizontals, distance):
        super().__init__(verticals, horizontals, distance)
        self.verticals = self.verticals - 1
        self.horizontals = self.horizontals - 1
        self.gridpoints = np.zeros((self.verticals * self.horizontals, 3),
                                   np.float32)
        self.gridpoints[:, :2] = self.distance * np.mgrid[0:self.verticals, 0:self.horizontals].T.reshape(-1, 2)
    

class Circles(Target):
    def __init__(self, verticals, horizontals, distance):
        super().__init__(verticals, horizontals, distance)
        self.pattern_size = (self.verticals, self.horizontals)
        spacing = self.distance * np.sqrt(2)
        num_points = int(self.verticals * (self.horizontals - 1) / 2 + self.verticals * (self.horizontals + 1) / 2)
        self.gridpoints = np.zeros((num_points, 3))
        counter = 0
        for i in range(self.horizontals):
            if i % 2 == 0:
                for j in range(self.verticals):
                    point = np.array([2 * j, i, 0])
                    self.gridpoints[counter] = point
                    counter += 1
            else:
                for j in range(self.verticals):
                    point = np.array([2 * j + 1, i, 0])
                    self.gridpoints[counter] = point
                    counter += 1
        self.gridpoints = self.gridpoints * spacing


if __name__ == "__main__":
    cb = Checkerboard(8,11,15)
    a = cb.gridpoints
    b = a.T
    
    cg = Circles(4,7,8)
    c = cg.gridpoints
    d = c.T
    
    plt.close("all")
    ax = plt.axes()
    # ax = plt.axes(projection='3d')
    ax.scatter(-d[1],-d[0])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
