# Licensed under the BSD 3-Clause License
# Copyright (C) 2021 GeospaceLab (geospacelab)
# Author: Lei Cai, Space Physics and Astronomy, University of Oulu

__author__ = "Lei Cai"
__copyright__ = "Copyright 2021, GeospaceLab"
__license__ = "BSD-3-Clause License"
__email__ = "lei.cai@oulu.fi"
__docformat__ = "reStructureText"


import numpy
from scipy.interpolate import interp1d
import geospacelab.toolbox.utilities.pydatetime as dttool
import geospacelab.toolbox.utilities.pybasic as basic
import datetime


def numpy_array_join_vertical(arr1, arr2):

    ndim2 = arr2.ndim
    if ndim2 == 1:
        ashape = (1, arr2.shape[0])
        arr2 = arr2.reshape(ashape)

    if arr1 is None:
        arr1 = numpy.empty((0, arr2.shape[1]))
    else:
        ndim1 = arr1.ndim
        if ndim1 == 1:
            ashape = (1, arr1.shape[0])
            arr1 = arr1.reshape(ashape)
    newarr = numpy.vstack((arr1, arr2))
    return newarr


def numpy_array_join_horizontal(arr1, arr2):
    ndim2 = arr2.ndim
    if ndim2 == 1:
        ashape = (arr2.shape[0], 1)
        arr2 = arr2.reshape(ashape)

    if arr1 is None:
        arr1 = numpy.empty((arr2.shape[0], 0))
    else:
        ndim1 = arr1.ndim
        if ndim1 == 1:
            ashape = (arr1.shape[0], 1)
            arr1 = arr1.reshape(ashape)
    newarr = numpy.hstack((arr1, arr2))
    return newarr


def numpy_array_self_mask(data, conditions=None):
    # conditions should be a list
    if conditions is None:
        return data
    for condition in conditions:
        if isinstance(condition, (float, int, str)):
            data = numpy.ma.masked_where(data == condition, data)
        elif isinstance(condition, dict):
            for key, value in condition.items():
                if key == 'between':
                    data = numpy.ma.masked_where((data >= value[0]) & (data <= value[1]), data)
                elif key == 'out':
                    data = numpy.ma.masked_where((data < value[0]) | (data > value[1]), data)
                elif key == 'greater':
                    data = numpy.ma.masked_where(data > value, data)
                elif key == 'smaller':
                    data = numpy.ma.masked_where(data < value, data)
                elif key == 'eq':
                    if not isinstance(value, list):
                        value = [value]
                    for v in value:
                        data = numpy.ma.masked_where(data == value, data)
                elif key == 'neq':
                    data = numpy.ma.masked_where(data != value, data)
    return data


def data_resample(
        x=None, y=None, xtype=None, xres=None, xresscale=1.1,
        method='Null',  # Null - insert NaN, 'linear', 'cubic', ... (interpolation method)
        axis=0, forward=True, depth=0.
):

    x1 = x
    if xtype == 'datetime':
        # dt0 = datetime.datetime(1970, 1, 1)
        sectime, dt0 = dttool.convert_datetime_to_sectime(x1)
        x1 = sectime

    diff_x1 = numpy.diff(x1.flatten())
    if xres is None:
        xres = numpy.median(diff_x1)
    inds = numpy.where(diff_x1 > xres * xresscale)[0]

    if len(inds) == 0:
        return x, y
    if depth > 10:
        return x, y

    inds = [i+1 for i in inds]
    # for x
    value = []
    for ind in inds:
        res = xres
        if xtype == 'datetime':
            res = datetime.timedelta(seconds=xres)
        if forward:
            value.append(x[ind - 1] + res/10)
        else:
            value.append(x[ind] - res/10)
    xnew = numpy.insert(x, inds, value, axis=axis)

    # for y
    if method == 'Null':
        value = numpy.nan
        ynew = numpy.insert(y, inds, value, axis=axis)
    else:
        ifunc = interp1d(x1.flatten(), y, kind=method, axis=axis)
        x_p = xnew
        if xtype=='datetime':
            x_p, dt0 = dttool.convert_datetime_to_sectime(xnew)
        ynew = ifunc(x_p.flatten())

    if method == 'Null':
        xnew, ynew = data_resample(
        x=xnew, y=ynew, xtype=xtype, xres=xres, xresscale=xresscale,
        method='Null',  # Null - insert NaN, 'linear', 'cubic', ... (interpolation method)
        axis=axis, forward=False, depth=depth+1
)

    return xnew, ynew


def regridding_2d_xgaps(
    x, y, z,
    xtype=None, xres=None):
    
    x1 = x.flatten()
    if xtype == 'datetime':
        # dt0 = datetime.datetime(1970, 1, 1)
        sectime, dt0 = dttool.convert_datetime_to_sectime(x1)
        x1 = sectime
    
    diff_x = numpy.diff(x1)
    if xres is None:
        xres = numpy.median(diff_x)
        
    if xtype == 'datetime' and basic.isnumeric(xres):
        xres = datetime.timedelta(seconds=xres)
        
    xx = numpy.hstack(((x.flatten() - xres / 2)[:, numpy.newaxis], (x.flatten() + xres / 2)[:, numpy.newaxis]))
    xnew = xx.flatten()
    
    if len(y.shape) == 1 or 1 in y.shape:
        ynew = y
    else:
        yy = numpy.hstack((y, y))
        ynew = yy.reshape((xnew.shape[0], y.shape[1]))
    
    zz = numpy.hstack((z, z))
    znew = zz.reshape((xnew.shape[0], z.shape[1]))
    znew[1::2, :] = numpy.nan
    znew = znew[:-1, :]
    
    return xnew, ynew, znew
         

def data_resample_2d(
        x=None, y=None, z=None, xtype=None, xres=None, xresscale=3,
        method='Null',  # Null - insert NaN, 'linear', 'cubic', ... (interpolation method)
        axis=0, forward=True, depth=0
):

    x1 = x
    if xtype == 'datetime':
        # dt0 = datetime.datetime(1970, 1, 1)
        sectime, dt0 = dttool.convert_datetime_to_sectime(x1)
        x1 = sectime

    diff_x1 = numpy.diff(x1.flatten())
    if xres is None:
        xres = numpy.median(diff_x1)
    inds = numpy.where(diff_x1 > xres * xresscale)[0]

    if len(inds) == 0:
        return x, y, z

    if depth > 10:
        return x, y, z

    inds = [i+1 for i in inds]
    # for x
    value = []
    for ind in inds:
        res = xres
        if xtype == 'datetime':
            res = datetime.timedelta(seconds=xres)
        if forward:
            value.append(x[ind - 1] + res/10)
        else:
            value.append(x[ind] - res/10)
    xnew = numpy.insert(x, inds, value, axis=axis)

    # for y
    method_y = 'nearest'
    ifunc = interp1d(x1.flatten(), y, kind=method_y, axis=axis, fill_value="extrapolate")
    x_p = xnew
    if xtype == 'datetime':
        x_p, dt0 = dttool.convert_datetime_to_sectime(xnew)
    ynew = ifunc(x_p.flatten())

    # for z
    if method == 'Null':
        value = numpy.nan
        znew = numpy.insert(z, inds, value, axis=axis)
    else:
        ifunc = interp1d(x1.flatten(), z, kind=method, axis=axis)
        x_p = xnew
        if xtype == 'datetime':
            x_p, dt0 = dttool.convert_datetime_to_sectime(xnew)
        znew = ifunc(x_p.flatten())
    if method == 'Null':
        xnew, ynew, znew = data_resample_2d(
            x=xnew, y=ynew, z=znew, xtype=xtype, xres=xres, xresscale=xresscale,
            method='Null',  # Null - insert NaN, 'linear', 'cubic', ... (interpolation method)
            axis=axis, forward=False, depth=depth+1
        )

    return xnew, ynew, znew
