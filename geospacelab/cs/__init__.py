# Licensed under the BSD 3-Clause License
# Copyright (C) 2021 GeospaceLab (geospacelab)
# Author: Lei Cai, Space Physics and Astronomy, University of Oulu

__author__ = "Lei Cai"
__copyright__ = "Copyright 2021, GeospaceLab"
__license__ = "BSD-3-Clause License"
__email__ = "lei.cai@oulu.fi"
__docformat__ = "reStructureText"


from geospacelab.cs._cs_base import SphericalCoordinates, CartesianCoordinates, SpaceCartesianCS, SpaceSphericalCS
from geospacelab.cs._geo import *
from geospacelab.cs._aacgm import AACGM
from geospacelab.cs._apex import APEX
# from geopack import geopack
from geospacelab.wrapper.geopack.geopack import geopack

GEOC=GEOCSpherical


def set_cs(name=None, **kwargs):
    kind = kwargs.pop('kind', None)
    if name.upper() == 'GEO':
        cls = GEO
    elif name.upper() == 'AACGM':
        cls = AACGM
    elif name.upper() == 'APEX':
        cls = APEX
    elif name.upper() == 'GEOD':
        cls = GEOD
    elif name.upper() == 'GEOC':
        if kind == 'sph':
            cls = GEOCSpherical
        else:
            cls = GEOCCartesian
    elif name.upper() == 'LENU':
        if kind == 'sph':
            cls = LENUSpherical
        else:
            cls = LENUCartesian
    else:
        raise NotImplementedError

    return cls(**kwargs)