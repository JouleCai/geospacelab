from geospacelab.cs._cs_base import SphericalCoordinates, CartesianCoordinates, SpaceCartesianCS, SpaceSphericalCS
from geospacelab.cs._geo import *
from geospacelab.cs._aacgm import AACGM
from geospacelab.cs._apex import APEX
from geopack import geopack


def set_cs(name=None, **kwargs):
    kind = kwargs.get('kind', None)
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