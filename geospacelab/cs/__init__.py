from geospacelab.cs._cs_base import SphericalCoordinates, CartesianCoordinates, SpaceCSBase
from geospacelab.cs._geo import GEO, GEOD, GEOC
from geospacelab.cs._aacgm import AACGM
from geospacelab.cs._apex import APEX
from geopack import geopack


def set_cs(name=None, coords=None, kind=None, ut=None, **kwargs):
    if name.upper() == 'GEO':
        cls = GEO
    elif name.upper() == 'AACGM':
        cls = AACGM
    elif name.upper() == 'APEX':
        cls = APEX
    elif name.upper() == 'GEOD':
        cls = GEOD
    elif name.upper() == 'GEOC':
        cls = GEOC
    else:
        raise NotImplementedError

    return cls(name=name.upper(), coords=coords, kind=kind, ut=ut, **kwargs)