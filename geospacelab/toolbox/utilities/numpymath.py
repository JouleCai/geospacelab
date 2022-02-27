# Licensed under the BSD 3-Clause License
# Copyright (C) 2021 GeospaceLab (geospacelab)
# Author: Lei Cai, Space Physics and Astronomy, University of Oulu

__author__ = "Lei Cai"
__copyright__ = "Copyright 2021, GeospaceLab"
__license__ = "BSD-3-Clause License"
__email__ = "lei.cai@oulu.fi"
__docformat__ = "reStructureText"


import numpy as np
from numpy.polynomial import Polynomial


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


def calc_curve_tangent_slope(x, y, degree=3, unit='radians'):
    poly = Polynomial.fit(x, y, degree)
    c = poly.convert().coef
    c_deriv = [i * cd for i, cd in enumerate(c)]
    c_deriv = c[1:]
    f_deriv = Polynomial(c_deriv)
    tan_slope = f_deriv(x)

    slope = np.arctan(tan_slope)

    if unit == 'degree':
        slope = slope / np.pi * 180.

    return slope


if __name__ == "__main__":
    x1 = np.array(range(10))
    y1 = np.array(range(10)) * 5
    calc_curve_tangent_slope(x1, y1)
