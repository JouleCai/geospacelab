import numpy as np
import scipy.interpolate as si


import geospacelab.toolbox.utilities.pylogging as mylog


def _remove_nan(func):
    def wrapper(*args, **kwargs):
        remove_nan = kwargs.pop('remove_nan', True)
        if not remove_nan:
            return func(*args, **kwargs)
        x = args[0]
        y = args[1]
        inds_valid = np.where(~np.isnan(x) & ~np.isnan(y))[0]
        if inds_valid.size < 2:
            mylog.StreamLogger.warning("Not enough valid data points for interpolation after removing NaN. Returning NaN.")
            return np.full_like(args[2], np.nan)
        x_valid = x[inds_valid]
        y_valid = y[inds_valid]
        yi = func(x_valid, y_valid, *args[2:], **kwargs)
        return yi
    return wrapper


def _segment(func):
    def wrapper(*args, **kwargs):
        segment = kwargs.pop('segment', False)
        extrap = kwargs.pop('extrap', False)
        extrap_boundary = kwargs.pop(
            'extrap_boundary', 'sharp'
        )    # 'sharp', 'soft' or 'outermost', only valid when extrap is True.
        x = args[0]
        y = args[1]
        xi = args[2]
        x_res = kwargs.pop('x_res', None)
        x_res_scale = kwargs.pop('x_res_scale', 1.)
        diff_x = np.diff(x)
        if x_res is None:
            x_res = np.nanmedian(diff_x)
        if not segment:
            yi_out = func(*args, **kwargs)
        else:
            yi_out = np.full_like(xi, np.nan)
            # Find the boundaries of valid segments
            inds_seg = np.where(diff_x > x_res * x_res_scale)[0]
            if len(inds_seg) == 0:
                yi_out = func(x, y, xi, *args[3:], **kwargs)
            else:
                segment_boundaries = inds_seg + 1
                segment_indices = np.split(np.arange(len(x)), segment_boundaries)
                for inds in segment_indices:
                    x_segment = x[inds]
                    y_segment = y[inds]
                    yi_segment = func(x_segment, y_segment, xi, *args[3:], **kwargs)
                    if yi_segment is None:
                        continue
                    if not extrap or extrap_boundary == 'sharp':
                        inds_valid = np.where((xi >= x_segment[0]) & (xi <= x_segment[-1]))[0]
                    else:
                        if extrap_boundary in ['soft', 'outermost']:
                            x_min = x_segment[0] - x_res * x_res_scale / 2
                            x_max = x_segment[-1] + x_res * x_res_scale / 2
                        else:
                            raise ValueError(f"Invalid extrapolation boundary: {extrap_boundary}")
                        inds_valid = np.where((xi >= x_min) & (xi <= x_max))[0]
                    yi_out[inds_valid] = yi_segment[inds_valid]
        if yi_out is None:
            return np.full_like(xi, np.nan)
        if not extrap:
            yi_out = np.where((xi < x[0]) | (xi > x[-1]), np.nan, yi_out)
        else:
            if extrap_boundary == 'sharp':
                x_min = x[0]
                x_max = x[-1]
            elif extrap_boundary == 'soft':
                x_min = x[0] - x_res * x_res_scale / 2
                x_max = x[-1] + x_res * x_res_scale / 2
            elif extrap_boundary == 'outermost':
                x_min = xi[0]
                x_max = xi[-1]
            else:
                raise ValueError(f"Invalid extrapolation boundary: {extrap_boundary}")
            yi_out = np.where((xi < x_min) | (xi > x_max), np.nan, yi_out)
        return yi_out
    return wrapper

@_remove_nan
@_segment
def interp1d_linear(x, y, xi, left=None, right=None, period=None, **kwargs):
    if len(x) < 2:
        mylog.StreamLogger.warning("Linear interpolation requires at least 2 data points. Falling back to nearest neighbor interpolation.")
        return None
    yi = np.interp(xi, x, y, left=left, right=right, period=period)
    return yi


@_remove_nan
@_segment
def interp1d_nearest(x, y, xi, **kwargs):
    """Interpolate 1-D data using the nearest neighbor method.

    Parameters
    ----------
    x : array_like
        The x-coordinates of the data points, must be increasing.
    y : array_like
        The y-coordinates of the data points.
    xi : array_like
        The x-coordinates at which to evaluate the interpolated values.

    Returns
    -------
    ndarray
        Interpolated values at xi.
    """
    if len(x) < 2:
        mylog.StreamLogger.warning("Nearest neighbor interpolation requires at least 2 data points. Falling back to linear interpolation.")
        return None
    diff_x = np.diff(x)
    x = x + np.hstack([diff_x/2, diff_x[-1]/2])
    # Append the last point in y twice for ease of use
    y = np.hstack([y, y[-1]])
    return y[np.searchsorted(x, xi)]

@_remove_nan
@_segment
def interp1d_cubic(x, y, xi, axis=0, bc_type='not-a-knot', extrapolate=None, **kwargs):
    """Interpolate 1-D data using cubic spline interpolation. A wrapper of scipy.interpolate.CubicSpline.

    Parameters
    ----------
    x : array_like
        The x-coordinates of the data points, must be increasing.
    y : array_like
        The y-coordinates of the data points.
    xi : array_like
        The x-coordinates at which to evaluate the interpolated values.

    Returns
    -------
    ndarray
        Interpolated values at xi.
    """
    if len(x) < 4:
        mylog.StreamLogger.warning("Cubic spline interpolation requires at least 4 data points. Falling back to linear interpolation.")
        return None
    f = si.CubicSpline(x, y, axis=axis, bc_type=bc_type, extrapolate=extrapolate)
    yi = f(xi)
    return yi 

@_remove_nan
@_segment
def interp1d_BSpline(x, y, xi, k=3, t=None, bc_type=None, axis=0, check_finite=True, **kwargs):
    """Interpolate 1-D data using B-Spline interpolation. A wrapper of scipy.interpolate.BSpline.

    Parameters
    ----------
    x : array_like
        The x-coordinates of the data points, must be increasing.
    y : array_like
        The y-coordinates of the data points.
    xi : array_like
        The x-coordinates at which to evaluate the interpolated values.

    Returns
    -------
    ndarray
        Interpolated values at xi.
    """
    if len(x) < k + 1:
        mylog.StreamLogger.warning(f"B-Spline interpolation of order {k} requires at least {k + 1} data points. Falling back to linear interpolation.")
        return None
    f = si.make_interp_spline(x, y, k=k, t=t, bc_type=bc_type, axis=axis, check_finite=check_finite)
    yi = f(xi)
    return yi


def interp1d(x, y, xi, method='linear', **kwargs):
    """Interpolate 1-D data. A wrapper of scipy.interpolate.interp1d with the handling of NaN and segment interpolation.

    Parameters
    ----------
    x : array_like
        The x-coordinates of the data points, must be increasing.
    y : array_like
        The y-coordinates of the data points.
    xi : array_like
        The x-coordinates at which to evaluate the interpolated values.
    method : str, optional
        The interpolation method. Options are 'linear', 'nearest', 'cubic', and 'BSpline'. Default is 'linear'.

    Returns
    -------
    ndarray
        Interpolated values at xi.
    """

    if method == 'linear':
        yi = interp1d_linear(x, y, xi, **kwargs)
    elif method == 'nearest':
        yi = interp1d_nearest(x, y, xi, **kwargs)
    elif method == 'cubic':
        yi = interp1d_cubic(x, y, xi, **kwargs)
    elif method == 'BSpline':
        yi = interp1d_BSpline(x, y, xi, **kwargs)
    else:
        raise ValueError(f"Unsupported interpolation method: {method}")
    
    return yi

def interp1d_periodic_y(
        x, y, xi, period=360., method='linear',
        segment_interp=True, segment_interp_extrapolate=False,
        x_res=None, x_res_scale=1., **kwargs):
    """Interpolate 1-D periodic data. A wrapper of scipy.interpolate.interp1d with the handling of periodicity.

    Parameters
    ----------
    x : array_like
        The x-coordinates of the data points, must be increasing.
    y : array_like
        The y-coordinates of the data points.
    xi : array_like
        The x-coordinates at which to evaluate the interpolated values.
    period : float, optional
        The period of the data. Default is 360.

    Returns
    -------
    ndarray
        Interpolated values at xi.
    """

    factor = 2 * np.pi / period
    sin_y = np.sin(y * factor)
    cos_y = np.cos(y * factor)
    # print(x, y)
    yi_sin = interp1d(
        x, sin_y, xi,
        method=method,
        segment_interp=segment_interp, 
        segment_interp_extrapolate=segment_interp_extrapolate,
        x_res=x_res, x_res_scale=x_res_scale, **kwargs)

    if all(np.isnan(yi_sin)):
        return np.full_like(xi, np.nan)

    yi_cos = interp1d(
        x, cos_y, xi, 
        method=method, 
        segment_interp=segment_interp, 
        segment_interp_extrapolate=segment_interp_extrapolate,
        x_res=x_res, x_res_scale=x_res_scale, **kwargs)

    if all(np.isnan(yi_cos)):
        return np.full_like(xi, np.nan)
    
    yi_sin = np.where(np.abs(yi_sin) > 1, yi_sin / np.sqrt(yi_sin**2 + yi_cos**2), yi_sin)
    yi_cos = np.where(np.abs(yi_cos) > 1, yi_cos / np.sqrt(yi_sin**2 + yi_cos**2), yi_cos)
    
    yi = np.arctan2(yi_sin, yi_cos) % (2 * np.pi) / factor

    return yi
    
    
if __name__ == '__main__':
    x = np.array([0, 0.5, 1, 1.5, 2, 3, 9, 20 , 21, 22])
    y = np.array([0, 45, 90, 135, 180,  225, 270, 315, 360, 405])
    xi = np.array([0.5, 1.2, 2.5, 3.5, 8.5, 9.5, 18.5, 20.5, 21.5])
    yi = interp1d_periodic_y(x, y, xi)
    print(yi)