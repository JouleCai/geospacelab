import datetime
import numpy as np

from geospacelab.cs._cs_base import SpaceCSBase, SphericalCoordinates, CartesianCoordinates
import geospacelab.toolbox.utilities.pylogging as mylog


class APEX(SpaceCSBase):
    def __init__(self, coords=None, ut=None, **kwargs):
        coords_model = SphericalCoordinates('APEX', add_coords=['lat', 'lon', 'height', 'mlt'])
        coords_model.config(**{
            'lat':          None,
            'lat_unit':     'degree',
            'lon':          None,
            'lon_unit':     'degree',
            'height':       None,
            'height_unit':  'km',
            'mlt':          None,
            'mlt_unit':     'h',
        })
        super().__init__(name='APEX', coords=coords, ut=ut, sph_or_car='sph', **kwargs)