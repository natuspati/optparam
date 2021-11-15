# -*- coding: utf-8 -*-
"""
Created on Sun Nov 14 17:34:16 2021

@author: bekdulnm
"""

import numpy as np
from class_defs import *

def unpack(focal, mirror_inner, mirror_outer):
    # reflect focal off the inner mirror
    focal_inner = mirror_inner.reflect_point(focal)
    
    # reflect focal off the outer mirror
    focal_outer = mirror_outer.reflect_point(focal_inner)
    
    
