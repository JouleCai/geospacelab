# Licensed under the BSD 3-Clause License
# Copyright (C) 2021 GeospaceLab (geospacelab)
# Author: Lei Cai, Space Physics and Astronomy, University of Oulu

__author__ = "Lei Cai"
__copyright__ = "Copyright 2021, GeospaceLab"
__license__ = "BSD-3-Clause License"
__email__ = "lei.cai@oulu.fi"
__docformat__ = "reStructureText"


import datetime
import numpy as np

from geospacelab.cs._cs_base import SpaceCSBase, SphericalCoordinates, CartesianCoordinates
import geospacelab.toolbox.utilities.pylogging as mylog


class APEX(SpaceCSBase):
    def __init__(self, coords=None, ut=None, **kwargs):
        kwargs.setdefault('new_coords', ['lat', 'lon', 'height', 'r', 'mlt'])
        super().__init__(name='APEX', coords=coords, ut=ut, kind='sph', **kwargs)
