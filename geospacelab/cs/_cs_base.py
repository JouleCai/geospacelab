import geospacelab.toolbox.utilities.pylogging as mylog
import geospacelab.toolbox.utilities.pyclass as pyclass


class SpaceCoordinateSystem(object):
    def __init__(self, name=None, coords=None, ut=None, sph_or_car=None, **kwargs):

        self.name = name
        self.ut = ut
        self.sph_or_car = sph_or_car
        self.coords = coords

    def transform(self, cs_to=None, **kwargs):
        if cs_to.lower() == 'aacgm':
            cs_new = self.to_aacgm(**kwargs)
        elif cs_to.lower() == 'apex':
            cs_new = self.to_apex(**kwargs)
        else:
            raise NotImplementedError
        return cs_new

    def config(self, logging=True, **kwargs):
        pyclass.set_object_attributes(self, append=False, logging=logging, **kwargs)

    @property
    def coords(self):
        return self._coords

    @coords.setter
    def coords(self, c):
        self._coords = None
        if issubclass(c.__class__, Coordinates):
            self._coords = c
        elif type(c) is dict:
            if self._coords is None:
                if self.sph_or_car == 'sph':
                    self._coords = SphericalCoordinates(cs=self.name)
                elif self.sph_or_car == 'car':
                    self._coords = CartesianCoordinates(cs=self.name)
                else:
                    raise NotImplementedError
            for key, value in c.items():
                if not hasattr(self._coords, key):
                    self._coords.add_coord(key)
                setattr(self._coords, key, value)


class Coordinates(object):
    def __init__(self, cs, kind=None):
        self.cs = cs
        self.kind = kind

    def config(self, logging=True, **kwargs):
        pyclass.set_object_attributes(self, append=False, logging=logging, **kwargs)

    def add_coord(self, name, unit=None):
        setattr(self, name, None)
        setattr(self, name + '_unit', unit)


class SphericalCoordinates(Coordinates):
    def __init__(self, cs):
        super().__init__(cs, kind='sph')
        self.lat = None
        self.lat_unit = 'degree'
        self.lon = None
        self.lon_unit = 'degree'
        self.h = None
        self.h_unit = 'km'
        self.r = None
        self.r_unit = 'km'

    def __call__(self, **kwargs):
        self.config(**kwargs)

    def to_car(self):
        raise NotImplemented


class CartesianCoordinates(Coordinates):
    def __init__(self, cs):
        super().__init__(cs, kind='sph')
        self.x = None
        self.x_unit = 'km'
        self.y = None
        self.y_unit = 'km'
        self.z = None
        self.z_unit = 'km'

    def __call__(self, **kwargs):
        self.config(**kwargs)

    def to_sph(self):
        raise NotImplemented
