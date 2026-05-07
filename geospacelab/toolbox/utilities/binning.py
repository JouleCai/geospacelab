import numpy as np


def binning1d(x, y, x_bins, method='mean', **kwargs):
    """Bin 1-D data.

    Parameters
    ----------
    x : array-like
        The x-coordinates of the data points.
    y : array-like
        The y-coordinates of the data points.
    x_bins : array-like
        The edges of the bins. The length of x_bins should be greater than 1.
    method : str, optional
        The method to use for binning. Options are 'mean', 'median', 'sum', 'min', 'max', 'std', or a percentile string like '90%'. Default is 'mean'.

    Returns
    -------
    y_binned : ndarray
        The binned values corresponding to each bin defined by x_bins.
    """
    if method == 'mean':
        func = np.nanmean
    elif method == 'median':
        func = np.nanmedian
    elif method == 'sum':
        func = np.nansum
    elif method == 'min':
        func = np.nanmin
    elif method == 'max':
        func = np.nanmax
    elif method == 'std':
        func = np.nanstd
    elif '%' in method:
        try:
            percentile = float(method.strip('%'))
            func = lambda arr: np.percentile(arr, percentile)
        except ValueError:
            raise ValueError(f"Invalid percentile value in method: {method}")
    else:
        raise ValueError(f"Unsupported binning method: {method}")
    
    inds_bin = np.digitize(x, x_bins) - 1  # Get bin indices for each x
    y_binned = np.full(len(x_bins) - 1, np.nan)  # Initialize binned values with NaN

    for i in range(len(x_bins) - 1):
        inds_in_bin = np.where(inds_bin == i)[0]
        if len(inds_in_bin) > 0 and np.any(~np.isnan(y[inds_in_bin])):
            y_binned[i] = func(y[inds_in_bin])
    
    return y_binned


def binning1d_moving(x, y, x_0, x_1, bin_step, bin_res, method='mean', **kwargs):
    """Bin 1-D data.

    Parameters
    ----------
    x : array-like
        The x-coordinates of the data points.
    y : array-like
        The y-coordinates of the data points.
    x_0 : float
        The starting value of the x-axis for the moving bins.
    x_1 : float
        The ending value of the x-axis for the moving bins.
    bin_step : float
        The step size for moving the bins along the x-axis.
    bin_res : float
        The resolution of each moving bin.
    method : str, optional
        The method to use for binning. Options are 'mean', 'median', 'sum', 'min', 'max', 'std', or a percentile string like '90%'. Default is 'mean'.

    Returns
    -------
    y_binned : ndarray
        The binned values corresponding to each moving bin.
    """
    
    if int(bin_res / bin_step) != bin_res / bin_step :
        raise ValueError("bin_res should be greater than or equal to bin_step and an integer multiple of bin_step to ensure proper binning.")
    
    n_binning = int(bin_res / bin_step)
    
    for i in range(n_binning):
        x_bins = np.arange(x_0 + i * bin_step, x_1 + bin_res, bin_res)
        y_binned = binning1d(x, y, x_bins, method=method, **kwargs)
        if i == 0:
            y_binned_all = y_binned[:, np.newaxis]
        else:
            y_binned_all = np.hstack((y_binned_all, y_binned[:, np.newaxis]))
    
    y_binned_all = y_binned_all.flatten()
    
    return y_binned_all