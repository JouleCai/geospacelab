# Licensed under the BSD 3-Clause License
# Copyright (C) 2021 GeospaceLab (geospacelab)
# Author: Lei Cai, Space Physics and Astronomy, University of Oulu

__author__ = "Lei Cai"
__copyright__ = "Copyright 2021, GeospaceLab"
__license__ = "BSD-3-Clause License"
__email__ = "lei.cai@oulu.fi"
__docformat__ = "reStructureText"


import numpy as np

def trig_arctan_to_sph_lon(x, y):
    isscalar = False
    if np.isscalar(x) and np.isscalar(y):
        isscalar = True
        x = np.array([x])
        y = np.array([y])
    old_settings = np.seterr(divide='ignore')     
    angle = np.arctan(np.divide(y, x))
    angle = np.where(x<0, angle + np.pi, angle)
    angle = np.mod(angle, 2*np.pi)
    np.seterr(**old_settings)
    if isscalar:
        angle = angle[0]
    return angle

