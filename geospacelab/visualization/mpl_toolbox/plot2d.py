import numpy as np
import matplotlib.pyplot as plt


def pcolormesh(x=None, y=None, z=None, ax=None, **kwargs):

    if ax is None:
        ax = plt.gca()

    nrows = z.shape[0]
    ncols = z.shape[1]

    if x is None:
        x = np.array(range(ncols+1))
    if y is None:
        y = np.array(range(nrows+1))

    if x.ndim == 1 and y.ndim == 1:
        x, y = np.meshgrid(x, y, sparse=True)
    elif x.shape[0] == 1:
        x = np.repeat(x.flatten()[np.newaxis, :], y.shape[0], axis=0)
    elif y.shape[0] == 1:
        y = np.repeat(y.flatten()[:, np.newaxis], x.shape[0], axis=1)

    im = ax.pcolormesh(x, y, z, **kwargs)
    return im