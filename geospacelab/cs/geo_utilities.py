# Licensed under the BSD 3-Clause License
# Copyright (C) 2021 GeospaceLab (geospacelab)
# Author: Lei Cai, Space Physics and Astronomy, University of Oulu

__author__ = "Lei Cai"
__copyright__ = "Copyright 2021, GeospaceLab"
__license__ = "BSD-3-Clause License"
__email__ = "lei.cai@oulu.fi"
__docformat__ = "reStructureText"


import numpy as np


def convert_local_az_el_range_to_geo(lat_0, lon_0, height_0, az, el, beam_range=None, radians = False):
    if radians:
        rd = 1.
    else:
        rd = np.pi / 180.
    lat_0 = lat_0 * rd
    lon_0 = lon_0 * rd
    az = az * rd
    el = el * rd


