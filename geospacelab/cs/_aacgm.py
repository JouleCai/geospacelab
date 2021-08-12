import datetime
import numpy as np

from geospacelab.cs._cs_base import SpaceCoordinateSystem, SphericalCoordinates, CartesianCoordinates
import geospacelab.toolbox.utilities.pylogging as mylog


class AACGM(SpaceCoordinateSystem):
    def __init__(self, coords=None, ut=None, **kwargs):
        coords_model = SphericalCoordinates('AACGM', add_coords=['lat', 'lon', 'r', 'mlt'])
        coords_model.config(**{
            'lat':          None,
            'lat_unit':     'degree',
            'lon':          None,
            'lon_unit':     'degree',
            'r':            None,
            'r_unit':       'Re',
            'height':       None,
            'height_unit':  'km',
            'mlt':          None,
            'mlt_unit':     'h',
        })
        super().__init__(name='AACGM', coords=coords, ut=ut, sph_or_car='sph', **kwargs)