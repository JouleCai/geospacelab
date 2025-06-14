# Licensed under the BSD 3-Clause License
# Copyright (C) 2021 GeospaceLab (geospacelab)
# Author: Lei Cai, Space Physics and Astronomy, University of Oulu

__author__ = "Lei Cai"
__copyright__ = "Copyright 2021, GeospaceLab"
__license__ = "BSD-3-Clause License"
__email__ = "lei.cai@oulu.fi"
__docformat__ = "reStructureText"


import numpy
import numpy as np
from scipy.interpolate import interp1d, griddata
import geospacelab.toolbox.utilities.pydatetime as dttool
import geospacelab.toolbox.utilities.pybasic as basic
import geospacelab.toolbox.utilities.pylogging as mylog
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

# def resample_2d(
#         x_data, y_data, z_data,
#         x_data_type=None,
#         x_data_res=None,
#         x_data_res_scale=1.,
#         x_grid_res=None,
#         x_grid_res_scale=1,
#         x_grid_min=None,
#         x_grid_max=None,
#         along_x_interp=True,
#         along_x_interp_method='nearest',
#         along_x_binning=False,
#         y_data_res=None,
#         y_data_res_scale=1.,
#         y_grid_res=None,
#         y_grid_res_scale=1.,
#         y_grid_min=None,
#         y_grid_max=None,
#         along_y_interp=True,
#         along_y_interp_method='nearest',
#         along_y_binning=False,
# ):
#     if along_x_interp==True and along_x_binning==True:
#         mylog.StreamLogger.error('The keywords "along_x_interp" and "along_x_binning" cannot be True at the same time!')
#         raise ValueError
#     if along_y_interp==True and along_y_binning==True:
#         mylog.StreamLogger.error('The keywords "along_y_interp" and "along_y_binning" cannot be True at the same time!')
#         raise ValueError
#
#     num_x, num_y = z_data.shape
#     if x_data_type == 'datetime':
#         dt0 = dttool.get_start_of_the_day(numpy.nanmin(x_data.flatten()))
#         sectime, dt0 = dttool.convert_datetime_to_sectime(x_data, dt0=dt0)
#         x_data = sectime
#
#     # check x dim:
#     if len(x_data.shape) == 1:
#         x_dim = 1
#     elif len(x_data.shape) == 2:
#         if x_data.shape[1] == num_y:
#             x_dim = 2
#         else:
#             x_dim = 1
#     else:
#         raise ValueError
#     if x_dim == 1:
#         xd = np.tile(x_data.flatten(), (1, num_y))
#     else:
#         xd = x_data
#     min_x = np.nanmin(xd.flatten())
#     max_x = np.nanmax(xd.flatten())
#     # check y dim:
#     if len(y_data.shape) == 1:
#         y_dim = 1
#     elif len(y_data.shape) == 2:
#         if y_data.shape[1] == num_y:
#             y_dim = 2
#         else:
#             y_dim = 1
#     else:
#         raise ValueError
#     if y_dim == 1:
#         yd = np.tile(x_data.flatten(), (num_x, 1))
#     else:
#         yd = y_data
#     min_y = np.nanmin(yd.flatten())
#     max_y = np.nanmax(yd.flatten())
#
#     if x_data_res is None:
#         x_data_res_ = np.median(np.diff(xd[:, 0].flatten()))
#     else:
#         x_data_res_ = x_data_res
#     if y_data_res is None:
#         y_data_res_ = np.median(np.diff(yd[0, :].flatten()))
#     else:
#         y_data_res_ = y_data_res
#
#     if x_grid_res is None:
#         xx = xd[:, 0].flatten()
#         # along_x_interp=False
#         # along_x_binning=False
#         # x_grid_res=x_data_res
#     else:
#         if x_grid_min is None:
#             x_grid_min = np.floor((min_x / x_grid_res)) * x_grid_res
#         if x_grid_max is None:
#             x_grid_max = np.ceil((max_x / x_grid_res)) * x_grid_res
#         xx = np.arange(x_grid_min, x_grid_max+x_grid_res, x_grid_res)
#     if y_grid_res is None:
#         yy = yd[0, :].flatten()
#         # along_y_interp=False
#         # along_y_binning=False
#         # y_grid_res=y_data_res
#     else:
#         if y_grid_min is None:
#             y_grid_min = np.floor((min_y / y_grid_res)) * y_grid_res
#         if y_grid_max is None:
#             y_grid_max = np.ceil((max_y / y_grid_res)) * y_grid_res
#         yy = np.arange(y_grid_min, y_grid_max+y_grid_res, y_grid_res)
#
#     grid_x, grid_y = numpy.meshgrid(xx,yy)
#     grid_z = np.empty_like(grid_x)
#     grid_z[::] = np.nan
#
#     if along_y_interp:
#         grid_x_1 = np.empty_like((num_x, grid_z.shape[1])) * np.nan
#         grid_y_1 = np.empty_like((num_x, grid_z.shape[1])) * np.nan
#         grid_z_1 = np.empty_like((num_x, grid_z.shape[1])) * np.nan
#         for i in range(num_x):
#             x1 = xd[i, :].flatten()
#             y1 = yd[i, :].flatten()
#             z1 = z_data[i, :].flatten()
#
#             inds_finite = np.where(np.isfinite(x1) & np.isfinite(y1) & np.isfinite(z1))[0]
#             if not list(inds_finite):
#                 continue
#
#             f = interp1d(y1[inds_finite], x1[inds_finite],
#                 kind='nearest', bounds_error=False, fill_value=np.nan)
#             x_i = f(grid_y[i, :].flatten())
#             grid_x_1[i, :] = x_i
#
#             f = interp1d(y1[inds_finite], y1[inds_finite],
#                 kind='nearest', bounds_error=False, fill_value=np.nan)
#             y_i = f(grid_y[i, :].flatten())
#             grid_y_1[i, :] = y_i
#
#             f = interp1d(
#                 y1[inds_finite], z1[inds_finite],
#                 kind=along_y_interp_method, bounds_error=False, fill_value=np.nan)
#             z_i = f(grid_y[i, :].flatten())
#
#             grid_z_1[i, :] = z_i


    

def regridding_2d_xgaps(
    x, y, z,
    xtype=None, xres=None):
    
    x1 = x.flatten()
    if xtype == 'datetime':
        # dt0 = datetime.datetime(1970, 1, 1)
        sectime, dt0 = dttool.convert_datetime_to_sectime(x1)
        x1 = sectime
    
    diff_x = numpy.diff(x1)
    x_res_md = numpy.median(diff_x)
    if xres is None:
        xres = x_res_md
        
    if xtype == 'datetime' and basic.isnumeric(xres):
        xres = datetime.timedelta(seconds=xres)
        x_res_md = datetime.timedelta(seconds=x_res_md)

    xx_1 = np.empty_like(x)
    xx_2 = np.empty_like(x)
    for i, xxx in enumerate(x):
        if i == 0:
            xx_1[0] = xxx - np.min([datetime.timedelta(diff_x[0]), xres, x_res_md]) / 2
        else:
            xx_1[i] = xx_2[i-1] if xres > datetime.timedelta(seconds=diff_x[i-1]) else xxx - xres / 2
        
        if i == len(x) - 1:
            xx_2[i] = xxx + np.min([datetime.timedelta(diff_x[i-1]), xres, x_res_md]) / 2 
        else:
            xx_2[i] = xxx + np.min([datetime.timedelta(diff_x[i]), xres, x_res_md]) / 2  
    
    xx = numpy.hstack((xx_1[:, numpy.newaxis], xx_2[:, numpy.newaxis]))
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
