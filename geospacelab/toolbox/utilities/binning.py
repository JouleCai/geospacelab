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
        The method to use for binning. Options are 'mean', 'median', 'sum', 'min', 'max'. Default is 'mean'.

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
        if len(inds_in_bin) > 0:
            y_binned[i] = func(y[inds_in_bin])
    
    return y_binned
