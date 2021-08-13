import geospacelab.toolbox.utilities.pylogging as mylog
import geospacelab.toolbox.utilities.pyclass as pyclass
import weakref
import numpy as np
from geopack import geopack

default_coord_attrs = {
    'sph':  {
        'lat':  {'unit': 'degree'},
        'lon':  {'unit': 'degree'},
        'phi':  {'unit': 'radians'},
        'theta': {'unit': 'radians'},
        'height': {'unit': 'km'},
        'r':    {'unit': 'Re'},
        'mlt':  {'unit': 'h'},
        'lst':  {'unit': 'h'},
    },
    'car':  {
        'x':    {'unit': 'km'},
        'y':    {'unit': 'km'},
        'z':    {'unit': 'km'}
    }
}


class SpaceCSBase(object):
    def __init__(self, name=None, coords=None, ut=None, kind=None, new_coords=None, **kwargs):

        self.name = name
        self.ut = ut
        self.coords = None
        self.kind = kind
        self._set_coords(kind, new_coords=new_coords)
        
        attrs = {}

        if isinstance(coords, dict):
            attrs = coords
        elif issubclass(coords.__class__, self.coords.__class__):
            self.coords = coords

        for key, value in attrs.items():
            if hasattr(self.coords, key):
                setattr(self.coords, key, value)
            else:
                mylog.StreamLogger.warning(
                    'The coords have not the attr {}. Add a new coord using the method "add_coord"'.format(key)
                )

    def _set_coords(self, sph_or_car, new_coords=None):
        if sph_or_car == 'sph':
            self.coords = SphericalCoordinates(cs=self, new_coords=new_coords)
        elif self.sph_or_car == 'car':
            self.coords = CartesianCoordinates(cs=self, new_coords=new_coords)

    def transform(self, cs_to=None, **kwargs):
        func = getattr(self, 'to_' + cs_to.upper())
        cs_new = func(**kwargs)
        return cs_new

    def config(self, logging=True, **kwargs):
        pyclass.set_object_attributes(self, append=False, logging=logging, **kwargs)

    def __getitem__(self, key):
        return getattr(self.coords, key)

    def __setitem__(self, key, value):
        if hasattr(self.coords, key):
            setattr(self.coords, key, value)
        else:
            raise KeyError


class CoordinatesBase(object):
    def __init__(self, cs=None, kind=None, new_coords=None):
        self.cs = weakref.proxy(cs)
        self.kind = kind
        if isinstance(new_coords, list):
            for coord_name in new_coords:
                self.add_coord(coord_name)
    
    def __call__(self, **kwargs):
        self.config(**kwargs)

    def config(self, logging=True, **kwargs):
        pyclass.set_object_attributes(self, append=False, logging=logging, **kwargs)

    def add_coord(self, name, value=None, unit=None):
        if not hasattr(self, name):
            setattr(self, name, value)
            if unit is None:
                unit = default_coord_attrs[self.kind][name]['unit']
            setattr(self, name + '_unit', unit)
        else:
            raise AttributeError("The coord is existing!")


class SphericalCoordinates(CoordinatesBase):
    def __init__(self, cs=None, **kwargs):
        self._Re = 6371.2
        kwargs.setdefault('new_coords', ['r', 'theta', 'phi'])
        super().__init__(cs=cs, kind='sph', **kwargs)

    def __call__(self, **kwargs):
        self.config(**kwargs)

    def convert_thetaphi_to_latlon(self):

        if self.phi_unit == 'radians':
            factor = 180. / np.pi
        elif self.phi_unit == 'degree':
            factor = 1.
        else:
            raise NotImplementedError
        lat = 90. - self.theta * factor
        lon = self.phi * factor
        self.add_coord('lat', value=lat)
        self.add_coord('lon', value=lon)

    def convert_h_to_r(self):
        r = (self.height + self._Re) / self._Re
        self.add_coord('r', value=r)
        return r

    def convert_r_to_h(self):
        if self.r_unit == 'Re':
            factor = self._Re
        elif self.r_unit == 'km':
            factor = 1.
        else:
            raise NotImplementedError
        height = self.r * factor - self._Re
        self.add_coord('height', value=height)
        return height

    def convert_latlon_to_thetaphi(self):
        if self.lon_unit == 'radians':
            factor = 1.
        elif self.phi_unit == 'degree':
            factor = np.pi / 180.
        else:
            raise NotImplementedError
        theta = 90. - self.lat * factor
        phi = self.phi * factor
        self.add_coord('phi', value=phi)
        self.add_coord('theta', value=theta)

    def to_car(self):
        if self.cs.name in ['AACGM', 'APEX']:
            raise NotImplementedError

        if self.cs.name in ['GEO', 'GEOD']:
            cs_new = self.cs.to_GEOC
            r = cs_new.coords.r
            phi = cs_new.coords.phi
            theta = cs_new.coords.theta
            return cs_new.to_car

        from geospacelab.cs import set_cs
        cs_new = set_cs(name=self.cs.name, kind='car', ut=self.cs.ut)
        cs_new['x'] = self.r * np.sin(self.theta) * np.cos(self.phi)
        cs_new['y'] = self.r * np.sin(self.theta) * np.sin(self.phi)
        cs_new['z'] = self.r * np.cos(self.theta)

        return cs_new


class CartesianCoordinates(CoordinatesBase):
    def __init__(self, cs=None, **kwargs):
        self._Re = 6371.2       # Earth radians in km
        kwargs.setdefault('new_coords', ['x', 'y', 'z'])
        super().__init__(cs=cs, kind='car', **kwargs)

    def to_sph(self):
        import geospacelab.toolbox.utilities.numpymath as npmath
        from geospacelab.cs import set_cs
        # phi: longitude, theta: co-latitude, r: radial distance
        r = np.sqrt(self.x**2 + self.y**2 + self.z**2)
        theta = np.arccos(self.z / r)
        phi = npmath.trig_arctan_to_sph_lon(self.x, self.y)

        cs_new = set_cs(name=self.cs.name, kind='sph', ut=self.cs.ut)

        cs_new['r'] = r
        cs_new['phi'] = phi
        cs_new['theta'] = theta
        
        return cs_new
