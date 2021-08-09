import datetime
import numpy as np

from geospacelab.cs._cs_base import SpaceCoordinateSystem, SphericalCoordinates, CartesianCoordinates
import geospacelab.toolbox.utilities.pylogging as mylog


class APEX(SpaceCoordinateSystem):
    def __init__(self, coords=None, ut=None, **kwargs):
        super().__init__(name='APEX', coords=coords, ut=ut, sph_or_car='sph', **kwargs)