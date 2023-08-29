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
from scipy.interpolate import interp1d


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


def interp_period_data(x, y, xq,  period=360., method='linear', **kwargs):
    factor = 2 * np.pi / period
    x = np.array(x, dtype=np.float64)
    y1 = np.mod(y * factor, 2 * np.pi)
    sin_y = np.sin(y1)
    cos_y = np.cos(y1)

    inds = []
    values_x = []
    values_y = []
    sign_sin = np.sign(sin_y)
    sign_cos = np.sign(cos_y)
    diff_sign_sin = np.diff(sign_sin)
    diff_sign_cos = np.diff(sign_cos)

    ind1 = np.where(diff_sign_sin == 2)[0]
    if ind1.size > 0:
        inds.extend(ind1 + 1)
        value_y = 0
        values_y.extend([value_y]*len(ind1))
        values_x.extend(
            [
                x[i] + (x[i+1] - x[i]) * np.mod((value_y - y1[i]), 2*np.pi) / np.mod((y1[i+1] - y1[i]), 2*np.pi)
                for i in ind1
            ]
        )

    ind1 = np.where(diff_sign_sin == -2)[0]
    if ind1.size > 0:
        inds.extend(ind1 + 1)
        value_y = np.pi
        values_y.extend([value_y] * len(ind1))
        values_x.extend(
            [
                x[i] + (x[i + 1] - x[i]) * np.mod((value_y - y1[i]), 2 * np.pi) / np.mod((y1[i + 1] - y1[i]), 2 * np.pi)
                for i in ind1
            ]
        )

    ind1 = np.where(diff_sign_cos == -2)[0]
    if ind1.size > 0:
        inds.extend(ind1 + 1)
        value_y = np.pi / 2
        values_y.extend([value_y] * len(ind1))
        values_x.extend(
            [
                x[i] + (x[i + 1] - x[i]) * np.mod((value_y - y1[i]), 2 * np.pi) / np.mod((y1[i + 1] - y1[i]), 2 * np.pi)
                for i in ind1
            ]
        )

    ind1 = np.where(diff_sign_cos == 2)[0]
    if ind1.size > 0:
        inds.extend(ind1 + 1)
        value_y = np.pi / 2 * 3
        values_y.extend([value_y] * len(ind1))
        values_x.extend(
            [
                x[i] + (x[i + 1] - x[i]) * np.mod((value_y - y1[i]), 2 * np.pi) / np.mod((y1[i + 1] - y1[i]), 2 * np.pi)
                for i in ind1
            ]
        )

    if list(inds):
        x = np.insert(x, inds, np.array(values_x))
        y1 = np.insert(y1, inds, np.array(values_y))
    else:
        y1 = y1
    sin_y = np.sin(y1)
    cos_y = np.cos(y1)

    itpf_sin = interp1d(x, sin_y, kind=method, **kwargs)
    itpf_cos = interp1d(x, cos_y, kind=method, **kwargs)
    sin_y_i = itpf_sin(xq)
    cos_y_i = itpf_cos(xq)
    rad = np.sign(sin_y_i) * (np.pi / 2 - np.arcsin(cos_y_i))
    rad = np.mod(rad, 2*np.pi)
    yq = rad / factor
    return yq


if __name__ == "__main__":
    x_1 = np.array(range(10))
    y_1 = np.array(range(10)) * 5
    calc_curve_tangent_slope(x_1, y_1)
