import geospacelab.toolbox.utilities.pylogging as mylog
import geospacelab.toolbox.utilities.pyclass as pyclass
import weakref
import numpy as np

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
    def __init__(self, name=None, coords=None, ut=None, sph_or_car=None, add_coords=None, **kwargs):

        self.name = name
        self.ut = ut
        self.coords = None
        self._set_coords(sph_or_car, add_coords=add_coords)
        
        attrs = {}
        if isinstance(coords, dict):
            attrs = coords
        elif issubclass(coords.__class__, CoordinatesBase):
            attrs = pyclass.get_object_attributes(coords)

        for key, value in attrs.items():
            if hasattr(self.coords, key):
                self.coords[key] = value
            else:
                mylog.StreamLogger.warning(
                    'The coords have not the attr {}. Add a new coord using the method "add_coord"'.format(key)
                )

    def _set_coords(self, sph_or_car, add_coords):
        if sph_or_car == 'sph':
            if add_coords is None:
                add_coords = ['phi', 'theta', 'r']
            self.coords = SphericalCoordinates(add_coords=add_coords)
        elif self.sph_or_car == 'car':
            self.coords = CartesianCoordinates(add_coords=add_coords)

    def transform(self, cs_to=None, **kwargs):
        func = getattr(self, 'to_' + cs_to)
        cs_new = func(**kwargs)
        return cs_new

    def config(self, logging=True, **kwargs):
        pyclass.set_object_attributes(self, append=False, logging=logging, **kwargs)


class CoordinatesBase(object):
    def __init__(self, cs=None, kind=None, add_coords=None):
        self.cs = weakref.proxy(cs)
        self.kind = kind
        if isinstance(add_coords, list):
            for coord_name in add_coords:
                self.add_coord(coord_name)
    
    def __call__(self, **kwargs):
        self.config(**kwargs)

    def config(self, logging=True, **kwargs):
        pyclass.set_object_attributes(self, append=False, logging=logging, **kwargs)

    def add_coord(self, name):
        if not hasattr(self, name):
            setattr(self, name, None)
            setattr(self, name + '_unit', default_coord_attrs[self.kind][name]['unit'])
        else:
            raise AttributeError("The coord is existing!")

class SphericalCoordinates(CoordinatesBase):
    def __init__(self, cs=None, **kwargs):
        super().__init__(cs=cs, kind='sph', **kwargs)

    def __call__(self, **kwargs):
        self.config(**kwargs)

    def to_car(self):
        if self.cs.name in ['GEO', 'GEOD']:
            from geopack import geopack
            hgeod = self.height
            xmugeod = self.lat / 180. * np.pi
            r, theta = geopack.geodgeo(hgeod,xmugeod,  1)
            phi = self.lon / 180 * np.pi
        else:
            r = self.r 
            theta = self.theta
            phi = self.phi
    from geospacelab.cs import CartesianCoordinates
    coords_new = CartesianCoordinates()
    coords_new.x = r * np.sin(theta) * np.cos(phi)
    coords_new.y = r * np.sin(thete) * np.sin(phi)
    coords_new.z = r * np.cos(theta) 

class CartesianCoordinates(CoordinatesBase):
    def __init__(self, cs=None, **kwargs):
        self._Re = 6371.2       # Earth radians in km
        
        super().__init__(cs=cs, kind='sph', **kwargs)
        self.x = None
        self.x_unit = 'Re'
        self.y = None
        self.y_unit = 'Re'
        self.z = None
        self.z_unit = 'Re'

    def to_sph(self):
        import geospacelab.toolbox.utilities.numpymath as npmath
        # phi: longitude, theta: co-latitude, r: radial distance
        coords_new = SphericalCoordinates(add_coords=['phi', 'theta', 'r'])
        coords_new.phi_unit = 'radians'
        coords_new.theta_unit = 'radians'
        coords_new.r_unit = 'Re'
        r = np.sqrt(self.x**2 + self.y**2 + self.z**2)
        if self.x_unit == 'km':
            r = r / self._Re

        r_xy = np.sqrt(self.x**2 + self.y**2)
        theta = np.arccos(self.z / r)
        phi = npmath.trig_arctan_to_sph_lon(self.x, self.y)
        
        coords_new.r = r
        coords_new.phi = phi
        coords_new.theta = theta
        
        if self.cs.name in ['GEO', 'GEOD']
        
        self.cs.coords = coords_new
