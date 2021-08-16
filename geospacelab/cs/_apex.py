import datetime
import numpy as np

from geospacelab.cs._cs_base import SpaceCSBase, SphericalCoordinates, CartesianCoordinates
import geospacelab.toolbox.utilities.pylogging as mylog


class APEX(SpaceCSBase):
    def __init__(self, coords=None, ut=None, **kwargs):
        kwargs.setdefault('new_coords', ['lat', 'lon', 'r', 'mlt'])
        super().__init__(name='APEX', coords=coords, ut=ut, kind='sph', **kwargs)
