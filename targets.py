# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 09:54:22 2021

@author: bekdulnm
"""

import numpy as np
import matplotlib.pyplot as plt


class Target(object):
    def __init__(self, verticals, horizontals):
        self.verticals = verticals
        self.horizontals = horizontals
        self.gridpoints = []


class Checkerboard(Target):
    def __init__(self, verticals, horizontals, checker_size):
        super().__init__(verticals, horizontals)
        self.verticals = self.verticals - 1
        self.horizontals = self.horizontals - 1
        self.checker_size = checker_size
        self.gridpoints = np.zeros((self.verticals * self.horizontals, 3),
                                   np.float32)
        self.gridpoints[:,:2] = self.checker_size * \
            np.mgrid[0:self.verticals, 0:self.horizontals].T.reshape(-1,2)
    

class Circles(Target):
    def __init__(self, verticals, horizontals):
        super().__init__(verticals, horizontals)
        self.pattern_size = ([self.verticals, self.horizontals])
        self.gridpoints = np.zeros((self.verticals * self.horizontals, 3),
                                   np.float32)
        self.gridpoints[:,:2] = np.mgrid[0:self.verticals,
                                         0:self.horizontals].T.reshape(-1, 2)


# cb = Checkerboard(8,11,15)
# a = cb.gridpoints
# b = a.T

# cg = Circles(8,11)
# c = cg.gridpoints
# d = c.T

# plt.close("all")
# ax = plt.axes(projection='3d')
# ax.scatter(d[0],d[1],d[2])
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')


