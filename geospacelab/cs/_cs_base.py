# Licensed under the BSD 3-Clause License
# Copyright (C) 2021 GeospaceLab (geospacelab)
# Author: Lei Cai, Space Physics and Astronomy, University of Oulu

__author__ = "Lei Cai"
__copyright__ = "Copyright 2021, GeospaceLab"
__license__ = "BSD-3-Clause License"
__email__ = "lei.cai@oulu.fi"
__docformat__ = "reStructureText"


import sys
import inspect

import geospacelab.toolbox.utilities.pylogging as mylog
import geospacelab.toolbox.utilities.pyclass as pyclass
import weakref
import numpy as np
from geopack import geopack

default_coord_attrs = {
    'sph':  {
        'lat':  {'unit': 'deg'},
        'lon':  {'unit': 'deg'},
        'phi':  {'unit': 'rad'},
        'theta': {'unit': 'rad'},
        'height': {'unit': 'km'},
        'r':    {'unit': 'Re'},
        'mlt':  {'unit': 'h'},
        'lst':  {'unit': 'h'},
        'az':   {'unit': 'deg'},
        'el':   {'unit': 'deg'},
        'range': {'unit':'deg'},
    },
    'car':  {
        'x':    {'unit': 'Re'},
        'y':    {'unit': 'Re'},
        'z':    {'unit': 'Re'},
        'E':    {'unit': 'km'},
        'N':    {'unit': 'km'},
        'Up':   {'unit': 'km'},
        'Down': {'unit': 'km'},
        'Forward':  {'unit': 'km'},
        'Perp':     {'unit': 'km'},
    }
}


class SpaceCSBase(object):
    def __init__(self,  name=None, vector=None, coords=None, ut=None, kind=None, new_coords=None, **kwargs):
        self.name = name
        self.ut = ut
        self.coords = None
        self.vector = vector
        self.kind = kind
        self._set_coords(coords, new_coords=new_coords)

    def __call__(self, *args, cs_to=None, **kwargs):
        if not list(args):
            self.vector = None
        elif len(args) == 1:
            from geospacelab.datahub import VariableModel
            if issubclass(args.__class__, VariableModel):
                self.vector = args[0].value
            else:
                self.vector = args[0]
        elif len(args) == 3:
            self.vector = np.concatenate(args, axis=len(args[0].shape)+1)
        else:
            raise NotImplementedError

        transform_func = getattr(self, 'to_' + cs_to.upper())
        cs_new = transform_func(**kwargs)
        return cs_new

    def _set_coords(self, coords, new_coords=None):
        if self.kind == 'sph':
            self.coords = SphericalCoordinates(cs=self, new_coords=new_coords)
        elif self.kind == 'car':
            self.coords = CartesianCoordinates(cs=self, new_coords=new_coords)

        attrs = {}

        if isinstance(coords, dict):
            attrs = coords
        elif issubclass(coords.__class__, self.coords.__class__):
            self.coords = coords
        for key, value in attrs.items():
            if hasattr(self.coords, key):
                if 'unit' not in key:
                    value = validate_coord_value(value)
                setattr(self.coords, key, value)
            else:
                mylog.StreamLogger.warning(
                    'The coords have not the attr {}. Add a new coord using the method "add_coord"'.format(key)
                )

    def config(self, logging=True, **kwargs):
        pyclass.set_object_attributes(self, append=False, logging=logging, **kwargs)

    def __getitem__(self, key):
        return getattr(self.coords, key)

    def __setitem__(self, key, value):
        if hasattr(self.coords, key):
            setattr(self.coords, key, value)
        else:
            raise KeyError

    def to_GEO(self, cs_new=None, **kwargs):
        if self.name == 'GEO':
            return self
        if cs_new is None:
            raise NotImplementedError
        else:
            return cs_new

    def to_GEOC(self, cs_new=None, **kwargs):
        if self.name == 'GEOC':
            return self

        if cs_new is None:
            raise NotImplementedError
        else:
            return cs_new

    def to_AACGM(self, cs_new=None, **kwargs):
        if self.name == 'AACGM':
            return self
        if cs_new is None:
            raise NotImplementedError
        else:
            return cs_new

    def to_APEX(self, cs_new=None, **kwargs):
        if self.name == 'APEX':
            return self
        if cs_new is None:
            raise NotImplementedError
        else:
            return cs_new

    def to_GEI(self, cs_new=None, **kwargs):
        if self.name == 'GEI':
            return self
        if cs_new is None:
            raise NotImplementedError
        else:
            return cs_new

    def to_SM(self, cs_new=None, **kwargs):
        if self.name == 'SM':
            return self
        if cs_new is None:
            raise NotImplementedError
        else:
            return cs_new

    def to_GSW(self, cs_new=None, **kwargs):
        if self.name == 'GSW':
            return self
        if cs_new is None:
            raise NotImplementedError
        else:
            return cs_new

    def to_QD(self, cs_new=None, **kwargs):
        if self.name == 'QD':
            return self
        if cs_new is None:
            raise NotImplementedError
        else:
            return cs_new


class SpaceSphericalCS(SpaceCSBase):
    def __init__(self,  name=None, coords=None, vector=None, ut=None, new_coords=None, **kwargs):

        super().__init__(name=name, vector=vector, coords=coords, ut=ut, kind='sph', new_coords=new_coords, **kwargs)

    def to_cartesian(self, **kwargs):
        from geospacelab.cs import set_cs

        if self.name in ['AACGM', 'APEX']:
            mylog.StreamLogger.error("{} has the spherical coordinates only!".format(self.name))

        if self.name in ['GEO', 'GEOD']:
            mylog.StreamLogger.warning("{} is converted into geocentric coordinates!".format(self.name))
            cs_new = self.to_GEOC()
            return cs_new.sph_to_car()

        x = self.coords.r * np.sin(self.coords.theta) * np.cos(self.coords.phi)
        y = self.coords.r * np.sin(self.coords.theta) * np.sin(self.coords.phi)
        z = self.coords.r * np.cos(self.coords.theta)

        coords = {'x': x, 'y': y, 'z': z}

        vector = None
        if self.vector is not None:
            mylog.StreamLogger.warning("The tranformation for vectors have not been implemented!")

        cs_new = set_cs(name=self.name, coords=coords, vector=vector, kind='car', ut=self.ut, **kwargs)

        return cs_new


class SpaceCartesianCS(SpaceCSBase):
    def __init__(self,  name=None, coords=None, vector=None, ut=None, new_coords=None, **kwargs):

        super().__init__(name=name, vector=vector, coords=coords, ut=ut, kind='car', new_coords=new_coords, **kwargs)

    def to_spherical(self, **kwargs):
        from geospacelab.cs import set_cs
        import geospacelab.toolbox.utilities.numpymath as npmath

        # phi: longitude, theta: co-latitude, r: radial distance
        r = np.sqrt(self.coords.x ** 2 + self.coords.y ** 2 + self.coords.z ** 2)
        theta = np.arccos(self.coords.z / r)
        phi = npmath.trig_arctan_to_sph_lon(self.coords.x, self.coords.y)
        vector = None
        if self.vector is not None:
            mylog.StreamLogger.warning("The tranformation for vectors have not been implemented!")

        coords = {'theta': theta, 'phi': phi, 'r': r}
        cs_new = set_cs(name=self.name, coords=coords,  vector=vector, kind='sph', ut=self.ut, **kwargs)

        return cs_new


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
            value = validate_coord_value(value)
            setattr(self, name, value)
            if unit is None:
                unit = default_coord_attrs[self.kind][name]['unit']
            setattr(self, name + '_unit', unit)
        else:
            raise AttributeError("The coord is existing!")


class SphericalCoordinates(CoordinatesBase):
    def __init__(self, cs=None, **kwargs):
        self.Re = np.double(6371.2)
        kwargs.setdefault('new_coords', ['r', 'theta', 'phi'])
        super().__init__(cs=cs, kind='sph', **kwargs)


class CartesianCoordinates(CoordinatesBase):
    def __init__(self, cs=None, **kwargs):
        self.Re = np.double(6371.2)     # Earth rad in km
        kwargs.setdefault('new_coords', ['x', 'y', 'z'])
        super().__init__(cs=cs, kind='car', **kwargs)


def validate_coord_value(value):

    if np.issubdtype(type(value), np.number):
        value = value
    elif type(value) in (list, tuple):
        value = np.array(value)
    elif isinstance(value, np.ndarray):
        value = value
    elif value is not None:
        raise TypeError
    return value

# def check_cs_method(cls):
#     sph_methods = ["convert_latlon_to_thetaphi", "convert_thetaphi_to_latlon",
#                    "convert_height_to_r", "convert_r_to_height"]
#     car_methods = ["convert_xyz_to_E_N_Up"]
#
#     def list_func():
#         func_dict = {
#             name: obj
#             for name, obj in inspect.getmembers(sys.modules[__name__])
#             if (inspect.isfunction(obj)) and (obj.__module__ == __name__)
#         }
#         return func_dict
#
#     def bind(instance, method):
#         def binding_scope_fn(*args, **kwargs):
#             return method(instance, *args, **kwargs)
#
#         return binding_scope_fn
#
#     def wrapper(*args, **kwargs):
#         kind = kwargs.setdefault('kind', None)
#         name = kwargs.setdefault('name', None)
#         func_dict = list_func()
#         instance = cls(*args, **kwargs)
#         if kind == "sph":
#             for sph_method in sph_methods:
#                 if name in ["GEO", "GEOD"]:
#                     continue
#                 setattr(instance, sph_method, bind(instance, func_dict[sph_method]))
#         if kind == "car":
#             for car_method in car_methods:
#
#                 setattr(instance, car_method, bind(instance, func_dict[car_method]))
#         return instance
#     return wrapper


# @check_cs_method
# class SpaceCSBase(object):
#     def __init__(self, name=None, coords=None, ut=None, kind=None, new_coords=None, **kwargs):
#
#         self.name = name
#         self.ut = ut
#         self.coords = None
#         self.kind = kind
#         self._set_coords(new_coords=new_coords)
#
#         attrs = {}
#
#         if isinstance(coords, dict):
#             attrs = coords
#         elif issubclass(coords.__class__, self.coords.__class__):
#             self.coords = coords
#
#         for key, value in attrs.items():
#             if hasattr(self.coords, key):
#                 setattr(self.coords, key, value)
#             else:
#                 mylog.StreamLogger.warning(
#                     'The coords have not the attr {}. Add a new coord using the method "add_coord"'.format(key)
#                 )
#
#     def _set_coords(self, new_coords=None):
#         if self.kind == 'sph':
#             self.coords = SphericalCoordinates(cs=self, new_coords=new_coords)
#         elif self.kind == 'car':
#             self.coords = CartesianCoordinates(cs=self, new_coords=new_coords)
#
#     def transform(self, cs_to=None, **kwargs):
#         func = getattr(self, 'to_' + cs_to.upper())
#         cs_new = func(**kwargs)
#         return cs_new
#
#     def sph_to_car(self):
#         if self.kind == 'sph':
#             return self
#
#         if self.name in ['AACGM', 'APEX']:
#             mylog.StreamLogger.error("{} has the spherical coordinates only!".format(self.name))
#
#         if self.name in ['GEO', 'GEOD']:
#             mylog.StreamLogger.warning("{} will be converted into GEOC in advance!".format(self.name))
#             cs_new = self.to_GEOC
#             return cs_new.sph_to_car()
#
#         from geospacelab.cs import set_cs
#         cs_new = set_cs(name=self.name, kind='car', ut=self.ut, new_coords=['x', 'y', 'z'])
#         cs_new['x'] = self.r * np.sin(self.theta) * np.cos(self.phi)
#         cs_new['y'] = self.r * np.sin(self.theta) * np.sin(self.phi)
#         cs_new['z'] = self.r * np.cos(self.theta)
#
#         return cs_new
#
#     def car_to_sph(self):
#         if self.kind == 'car':
#             return self
#         if self.name in ['AACGM', 'APEX', 'GEO', 'GEOD']:
#             mylog.StreamLogger.error("{} has no Cartersian coordinates!".format(self.name))
#
#         import geospacelab.toolbox.utilities.numpymath as npmath
#         from geospacelab.cs import set_cs
#         # phi: longitude, theta: co-latitude, r: radial distance
#         r = np.sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2)
#         theta = np.arccos(self.z / r)
#         phi = npmath.trig_arctan_to_sph_lon(self.x, self.y)
#
#         cs_new = set_cs(name=self.name, kind='sph', ut=self.ut, new_coords=['r', 'theta', 'phi'])
#
#         cs_new['r'] = r
#         cs_new['phi'] = phi
#         cs_new['theta'] = theta
#
#         return cs_new
#
#     def config(self, logging=True, **kwargs):
#         pyclass.set_object_attributes(self, append=False, logging=logging, **kwargs)
#
#     def __getitem__(self, key):
#         return getattr(self.coords, key)
#
#     def __setitem__(self, key, value):
#         if hasattr(self.coords, key):
#             setattr(self.coords, key, value)
#         else:
#             raise KeyError
#
#
# class CoordinatesBase(object):
#     def __init__(self, cs=None, kind=None, new_coords=None):
#         self.cs = weakref.proxy(cs)
#         self.kind = kind
#         if isinstance(new_coords, list):
#             for coord_name in new_coords:
#                 self.add_coord(coord_name)
#
#     def __call__(self, **kwargs):
#         self.config(**kwargs)
#
#     def config(self, logging=True, **kwargs):
#         pyclass.set_object_attributes(self, append=False, logging=logging, **kwargs)
#
#     def add_coord(self, name, value=None, unit=None):
#         if not hasattr(self, name):
#             setattr(self, name, value)
#             if unit is None:
#                 unit = default_coord_attrs[self.kind][name]['unit']
#             setattr(self, name + '_unit', unit)
#         else:
#             raise AttributeError("The coord is existing!")
#
#
# class SphericalCoordinates(CoordinatesBase):
#     def __init__(self, cs=None, **kwargs):
#         self._Re = 6371.2
#         kwargs.setdefault('new_coords', ['r', 'theta', 'phi'])
#         super().__init__(cs=cs, kind='sph', **kwargs)
#
#     def _convert_height_to_r(self):
#         r = (self.height + self._Re) / self._Re
#         self.add_coord('r', value=r)
#         return r
#
#     def _convert_r_to_height(self):
#         if self.r_unit == 'Re':
#             factor = self._Re
#         elif self.r_unit == 'km':
#             factor = 1.
#         else:
#             raise NotImplementedError
#         height = self.r * factor - self._Re
#         self.add_coord('height', value=height)
#         return height
#
#     def _convert_latlon_to_thetaphi(self):
#         if self.lon_unit == 'rad':
#             factor = 1.
#         elif self.phi_unit == 'deg':
#             factor = np.pi / 180.
#         else:
#             raise NotImplementedError
#         theta = 90. - self.lat * factor
#         phi = self.phi * factor
#         self.add_coord('phi', value=phi)
#         self.add_coord('theta', value=theta)
#
#     def
#
#
# def convert_thetaphi_to_latlon(self):
#
#     if self.phi_unit == 'rad':
#         factor = 180. / np.pi
#     elif self.phi_unit == 'deg':
#         factor = 1.
#     else:
#         raise NotImplementedError
#     lat = 90. - self.theta * factor
#     lon = self.phi * factor
#     self.add_coord('lat', value=lat)
#     self.add_coord('lon', value=lon)
#
#
# def convert_height_to_r(self):
#     r = (self.height + self._Re) / self._Re
#     self.add_coord('r', value=r)
#     return r
#
#
# def convert_r_to_height(self):
#     if self.r_unit == 'Re':
#         factor = self._Re
#     elif self.r_unit == 'km':
#         factor = 1.
#     else:
#         raise NotImplementedError
#     height = self.r * factor - self._Re
#     self.add_coord('height', value=height)
#     return height
#
#
# def convert_latlon_to_thetaphi(self):
#     if self.lon_unit == 'rad':
#         factor = 1.
#     elif self.phi_unit == 'deg':
#         factor = np.pi / 180.
#     else:
#         raise NotImplementedError
#     theta = 90. - self.lat * factor
#     phi = self.phi * factor
#     self.add_coord('phi', value=phi)
#     self.add_coord('theta', value=theta)
#
#
# class CartesianCoordinates(CoordinatesBase):
#     def __init__(self, cs=None, **kwargs):
#         self._Re = 6371.2       # Earth rad in km
#         kwargs.setdefault('new_coords', ['x', 'y', 'z'])
#         super().__init__(cs=cs, kind='car', **kwargs)
#
#
