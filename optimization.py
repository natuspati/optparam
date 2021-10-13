#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 10:44:17 2021

@author: bekdulnm
"""


import numpy as np


if __name__ == '__main__':
    # load points
    targetpoints = np.loadtxt("targetpoints.csv", delimiter=",")
    imgpoints= np.loadtxt("imgpoints.csv", delimiter=",")
    