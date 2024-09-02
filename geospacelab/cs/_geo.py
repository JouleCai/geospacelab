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

from geospacelab.cs._cs_base import SpaceCSBase, SphericalCoordinates, CartesianCoordinates, SpaceCartesianCS, SpaceSphericalCS
import geospacelab.toolbox.utilities.pylogging as mylog
import geospacelab.toolbox.utilities.pybasic as pybasic


class GEO(SpaceSphericalCS):
    def __init__(self, coords=None, ut=None, vector=None, **kwargs):
        
        kwargs.setdefault('new_coords', ['lat', 'lon', 'height', 'lst'])
        super().__init__(name='GEO', coords=coords, vector=vector, ut=ut, **kwargs)

    def to_AACGM(self, append_mlt=False):
        cs_geoc = self.to_GEOC(kind='sph')
        cs_new = cs_geoc.to_AACGM(append_mlt=append_mlt)
        return cs_new

    def to_APEX(self, append_mlt=False):
        cs_geoc = self.to_GEOC(kind='sph')
        cs_new = cs_geoc.to_APEX(append_mlt=append_mlt)
        return cs_new

    def to_LENU(self, lat_0=None, lon_0=None, height_0=None, kind='car', **kwargs):
        """
        Convert GEO to local LENU
        :param lat_0: geographic latitude
        :param lon_0: geographic longitude
        :param height_0: altiude from the sea level
        :return: LENU object.
        """

        cs_0 = GEO(coords={'lat': lat_0, 'lon': lon_0, 'height': height_0})

        cs_geoc = self.to_GEOC(kind='car')
        cs_new = cs_geoc.to_LENU(lat_0=cs_0['lat'], lon_0=cs_0['lon'], height_0=cs_0['height'], kind=kind)

        return cs_new

    def to_GEOC(self, kind='sph', **kwargs):
        from geospacelab.cs import geopack
        height = self.coords.height
        if self['lat_unit'] == 'deg':
            factor = np.pi / 180.
        elif self['lat_unit'] == 'rad':
            factor = 1.
        else:
            raise AttributeError
        lat = self['lat'] * factor

        if np.isscalar(lat) and np.isscalar(height):
            r, theta = geopack.geodgeo(height, lat, 1)
        else:
            if np.isscalar(lat) and not np.isscalar(height):
                heights = height.flatten()
                lats = np.empty_like(heights)
                lats[:] = lat
                shape = height.shape
            elif not np.isscalar(lat) and np.isscalar(height):
                lats = lat.flatten()
                heights = np.empty_like(lats)
                heights[:] = height
                shape = lat.shape
            else:
                lats = lat.flatten()
                heights = height.flatten()
                shape = lat.shape

            rs = np.empty_like(lats)
            thetas = np.empty_like(lats)
            for ind, lat1 in enumerate(lats):
                height1 = heights[ind]
                rs[ind], thetas[ind] = geopack.geodgeo(height1, lat1, 1)
            r = np.reshape(rs, shape)
            theta = np.reshape(thetas, shape)

        r = r / self.coords.Re            # unit in Re
        phi = self['lon'] * factor

        coords = {'r': r, 'theta': theta, 'phi': phi}

        vector = self.vector

        cs_new = GEOCSpherical(coords=coords, ut=self.ut, vector=vector)

        if kind == 'car':
            cs_new = cs_new.to_cartesian()
        return cs_new


class GEOD(GEO):
    def __init__(self, coords=None, ut=None, vector=None, **kwargs):
        super().__init__(name='GEO', coords=coords, vector=vector, ut=ut, **kwargs)


class LENUCartesian(SpaceCartesianCS):
    def __init__(self, coords=None, lat_0=None, lon_0=None, height_0=None, ut=None, vector=None, **kwargs):
        kwargs.setdefault('new_coords', ['E', 'N', 'Up', 'x', 'y', 'z'])
        self.lat_0 = lat_0
        self.lon_0 = lon_0
        self.height_0 = height_0
        super().__init__(name='LENU', coords=coords, ut=ut, vector=vector, **kwargs)

        self.check_coords()

    def check_coords(self):
        if self.coords.E is not None and self.coords.x is None:
            self.convert_ENU_to_xyz()
        elif self.coords.E is None and self.coords.x is not None:
            self.convert_xyz_to_ENU()

    def convert_ENU_to_xyz(self):
        x = self.coords.E / self.coords.Re
        y = self.coords.N / self.coords.Re
        z = self.coords.Up / self.coords.Re
        self.coords.x = x
        self.coords.y = y
        self.coords.z = z

    def convert_xyz_to_ENU(self):
        E = self.coords.x * self.coords.Re
        N = self.coords.y * self.coords.Re
        Up = self.coords.z * self.coords.Re
        self.coords.E = E
        self.coords.N = N
        self.coords.Up = Up

    def to_GEOC(self, kind='sph', **kwargs):

        cs_0 = GEO(coords={'lat': self.lat_0, 'lon': self.lon_0, 'height': self.height_0})
        cs_0 = cs_0.to_GEOC(kind='sph')
        phi_0 = cs_0['phi']
        theta_0 = cs_0['theta']
        cs_0 = cs_0.to_cartesian()
        x_0 = cs_0['x']
        y_0 = cs_0['y']
        z_0 = cs_0['z']

        R_1 = np.array([
            [1,     0,                  0],
            [0,     np.cos(theta_0),    np.sin(theta_0)],
            [0,     -np.sin(theta_0),    np.cos(theta_0)]
        ])

        R_2 = np.array([
            [-np.sin(phi_0),    np.cos(phi_0),     0],
            [-np.cos(phi_0),     -np.sin(phi_0),     0],
            [0,                 0,                  1]
        ])

        x = self.coords.x
        y = self.coords.y
        z = self.coords.z
        shape_x = x.shape

        v = np.array([x.flatten(), y.flatten(), z.flatten()]).T
        v_0 = np.array([x_0, y_0, z_0])
        v_0 = np.tile(v_0, (x.size, 1))

        v_new = v @ R_1 @ R_2 + v_0

        x_new = np.reshape(v_new[:, 0], shape_x)
        y_new = np.reshape(v_new[:, 1], shape_x)
        z_new = np.reshape(v_new[:, 2], shape_x)

        coords = {'x': x_new, 'y': y_new, 'z': z_new}

        vector = self.vector
        if vector is not None:
            raise NotImplementedError

        cs_new = GEOCCartesian(coords=coords, vector=vector)

        if kind == 'sph':
            cs_new = cs_new.to_spherical()
        return cs_new

    def to_spherical(self, **kwargs):
        cs_new = super(LENUCartesian, self).to_spherical(lat_0=self.lat_0, lon_0=self.lon_0, height_0=self.height_0)
        return cs_new

    def to_GEO(self, **kwargs):
        cs_new = self.to_GEOC()
        cs_new = cs_new.to_GEO()
        return cs_new


class LENUSpherical(SpaceSphericalCS):
    def __init__(self, coords=None, lat_0=None, lon_0=None, height_0=None, ut=None, vector=None, **kwargs):
        kwargs.setdefault('new_coords', ['az', 'el', 'range', 'theta', 'phi', 'r'])

        self.lat_0 = lat_0
        self.lon_0 = lon_0
        self.height_0 = height_0
        super().__init__(name='LENU', coords=coords, ut=ut, vector=vector, **kwargs)

        self.check_coords()

    def check_coords(self):
        if self.coords.phi is None and self.coords.az is not None:
            self.convert_rangeazel_to_rphitheta()
        elif self.coords.phi is not None and self.coords.az is None:
            self.convert_rphitheta_to_rangeazel()

    def convert_rangeazel_to_rphitheta(self):
        if self.coords.az_unit == 'deg':
            factor = np.pi / 180.
        elif self.coords.az_unit == 'rad':
            factor = 1.
        else:
            raise NotImplemented
        phi = np.mod(np.pi/2 - self.coords.az * factor, 2*np.pi)
        theta = np.pi / 2 - self.coords.el * factor
        r = self.coords.range / self.coords.Re
        self.coords.phi = phi
        self.coords.theta = theta
        self.coords.r = r
        return theta, phi, r

    def convert_rphitheta_to_rangeazel(self):
        if self.coords.phi_unit == 'deg':
            factor = 1
        elif self.coords.phi_unit == 'rad':
            factor = 180. / np.pi
        else:
            raise NotImplemented
        az = np.mod(90. - self.coords.phi*factor, 360)
        el = 90. - self.coords.theta * factor
        range = self.coords.r * self.coords.Re

        self.coords.az = az
        self.coords.el = el
        self.coords.range = range
        return az, el, range

    def to_cartesian(self, **kwargs):
        cs_new = super(LENUSpherical, self).to_cartesian(lat_0=self.lat_0, lon_0=self.lon_0, height_0=self.height_0)
        return cs_new

    def to_GEO(self, **kwargs):
        cs_new = self.to_cartesian()
        cs_new = cs_new.to_GEO()
        return cs_new

    def to_GEOC(self, kind='sph', **kwargs):
        cs_new = self.to_cartesian()
        cs_new = cs_new.to_GEOC(kind=kind)
        return cs_new


class GEOCSpherical(SpaceSphericalCS):
    def __init__(self, coords=None, ut=None, vector=None, **kwargs):
        kwargs.setdefault('new_coords', ['phi', 'theta', 'r', 'lat', 'lon', 'height'])
        super().__init__(name='GEOC', coords=coords, ut=ut, vector=vector, **kwargs)

        self.check_coords()

    def check_coords(self):
        if self.coords.lat is not None and self.coords.theta is None:
            self.convert_latlon_to_thetaphi()
        elif self.coords.lat is None and self.coords.theta is not None:
            self.convert_thetaphi_to_latlon()
        if self.coords.height is None and self.coords.r is not None:
            self.convert_r_to_height()
        elif self.coords.height is not None and self.coords.r is None:
            self.convert_height_to_r()

    def convert_latlon_to_thetaphi(self):
        if self.coords.lon_unit == 'rad':
            factor = 1.
        elif self.coords.lat_unit == 'deg':
            factor = np.pi / 180.
        else:
            raise NotImplementedError
        theta = np.pi/2 - self.coords.lat * factor
        phi = self.coords.lon * factor
        self['phi'] = phi
        self['theta'] = theta
        return theta, phi

    def convert_thetaphi_to_latlon(self):

        if self.coords.phi_unit == 'rad':
            factor = 180. / np.pi
        elif self.coords.phi_unit == 'deg':
            factor = 1.
        else:
            raise NotImplementedError
        lat = 90. - self.coords.theta * factor
        lon = self.coords.phi * factor
        self['lat'] = lat
        self['lon'] = lon
        return lat, lon

    def convert_height_to_r(self):
        r = (self.coords.height + self.coords.Re) / self.coords.Re
        self['r'] = r
        return r

    def convert_r_to_height(self):
        if self.coords.r_unit == 'Re':
            factor = self.coords.Re
        elif self.coords.r_unit == 'km':
            factor = 1.
        else:
            raise NotImplementedError
        height = self.coords.r * factor - self.coords.Re
        self['height'] = height
        return height

    def to_LENU(self, kind='sph', **kwargs):
        cs_new = self.to_cartesian()
        return cs_new.to_LENU(kind=kind, **kwargs)

    def to_GEO(self, **kwargs):
        from geospacelab.cs import geopack

        r = self.coords.r * self.coords.Re
        theta = self.coords.theta

        if np.isscalar(r) and np.isscalar(theta):
            h, lat = geopack.geodgeo(r, theta, 1)
        else:
            if np.isscalar(theta) and not np.isscalar(r):
                rs = r.flatten()
                thetas = np.empty_like(rs)
                thetas[:] = theta
                shape = r.shape
            elif not np.isscalar(theta) and np.isscalar(r):
                thetas = theta.flatten()
                rs = np.empty_like(thetas)
                rs[:] = r
                shape = thetas.shape
            else:
                thetas = theta.flatten()
                rs = r.flatten()
                shape = theta.shape

            hs = np.empty_like(thetas)
            lats = np.empty_like(thetas)
            for ind, r1 in enumerate(rs):
                theta1 = thetas[ind]
                hs[ind], lats[ind] = geopack.geodgeo(r1, theta1, -1)
            h = np.reshape(hs, shape)
            lat = np.reshape(lats, shape)
        lon = self.coords.phi

        factor = 180. / np.pi

        coords = {'lat': lat*factor, 'lon': lon*factor, 'height': h}
        vector = self.vector
        # if vi is not None:
        #    raise NotImplementedError

        cs_new = GEO(coords=coords, vector=vector, ut=self.ut)
        return cs_new

    def to_AACGM(self, append_mlt=False, **kwargs):
        import aacgmv2 as aacgm
        from geospacelab.cs._aacgm import AACGM
        method_code = 'G2A'

        ut_type = type(self.ut)
        if ut_type is list:
            uts = np.array(self.ut)
        elif ut_type is np.ndarray:
            uts = self.ut
        lat_shape = self.coords.lat.shape
        lon_shape = self.coords.lon.shape
        if issubclass(self.ut.__class__, datetime.datetime):

            lat, lon, r = aacgm.convert_latlon_arr(
                in_lat=self.coords.lat.flatten(), in_lon=self.coords.lon.flatten(), height=self.coords.height.flatten(),
                dtime=self.ut, method_code=method_code
            )
        else:
            if uts.shape[0] != self.coords.lat.shape[0]:
                mylog.StreamLogger.error("Datetimes must have the same length as cs!")
                return
            lat = np.empty_like(self.coords.lat)
            lon = np.empty_like(self.coords.lon)
            r = np.empty_like(self.coords.lat)
            for ind_dt, dt in enumerate(uts.flatten()):
                # print(ind_dt, dt, cs.lat[ind_dt, 0])
                lat[ind_dt], lon[ind_dt], r[ind_dt] = aacgm.convert_latlon_arr(in_lat=self.coords.lat[ind_dt],
                                                                               in_lon=self.coords.lon[ind_dt],
                                                                               height=self.coords.height[ind_dt],
                                                                               dtime=dt, method_code=method_code)
        cs_new = AACGM(coords={'lat': lat.reshape(lat_shape),
                               'lon': lon.reshape(lon_shape),
                               'r': r.reshape(lat_shape), 'r_unit': 'R_E'},
                       ut=self.ut)
        if append_mlt:
            lon = lon.flatten()
            if issubclass(self.ut.__class__, datetime.datetime):
                mlt = aacgm.convert_mlt(lon, self.ut)
            else:
                lon = cs_new['lon']
                mlt = np.empty_like(lon)
                for ind_dt, dt in enumerate(self.ut.flatten()):
                    mlt[ind_dt] = aacgm.convert_mlt(lon[ind_dt], dt)
            cs_new['mlt'] = mlt.reshape(lon_shape)
        return cs_new

    def to_APEX(self, append_mlt=False, **kwargs):
        import apexpy as apex
        from geospacelab.cs._apex import APEX

        cs_in = self

        ut_type = type(cs_in.ut)
        if ut_type is list:
            uts = np.array(cs_in.ut)
        elif ut_type is np.ndarray:
            uts = cs_in.ut

        mlt = None
        if issubclass(self.ut.__class__, datetime.datetime):
            apex_obj = apex.Apex(cs_in.ut)
            mlat, mlon = apex_obj.convert(
                cs_in.coords.lat, cs_in.coords.lon, 'geo', 'apex', height=cs_in.coords.height,
            )
            if append_mlt:
                mlt = apex_obj.mlon2mlt(mlon, cs_in.ut)
        else:
            if uts.shape[0] != cs_in.coords.lat.shape[0]:
                mylog.StreamLogger.error("Datetimes must have the same length as cs!")
                return

            mlat = np.empty_like(cs_in.coords.lat)
            mlon = np.empty_like(cs_in.coords.lat)
            mlt = np.empty_like(cs_in.coords.lat)
            for ind_dt, dt in enumerate(uts.flatten()):
                # print(ind_dt, dt, cs.lat[ind_dt, 0])
                apex_obj = apex.Apex(dt)
                mlat[ind_dt], mlon[ind_dt] = apex_obj.convert(
                    cs_in.coords.lat[ind_dt], cs_in.coords.lon[ind_dt], 'geo', 'apex',
                    height=cs_in.coords.height[ind_dt]
                )
                if append_mlt:
                    mlt[ind_dt] = apex_obj.mlon2mlt(mlon[ind_dt], dt)

        cs_new = APEX(coords={'lat': mlat, 'lon': mlon, 'height': cs_in.coords.height, 'mlt': mlt}, ut=cs_in.ut)

        return cs_new


class GEOCCartesian(SpaceCartesianCS):
    def __init__(self, coords=None, ut=None, vector=None, **kwargs):
        kwargs.setdefault('new_coords', ['x', 'y', 'z'])

        super().__init__(name='GEOC', coords=coords, ut=ut, vector=vector, **kwargs)

    def to_GEO(self, **kwargs):
        cs_new = self.to_spherical()
        return cs_new.to_GEO()

    def to_AACGM(self, append_mlt=False, **kwargs):
        cs_new = self.to_spherical()
        return cs_new.to_AACGM(append_mlt=append_mlt, **kwargs)

    def to_APEX(self, append_mlt=False, **kwargs):
        cs_new = self.to_spherical()
        return cs_new.to_APEX(append_mlt=append_mlt, **kwargs)

    def to_LENU(self, lat_0=None, lon_0=None, height_0=None, kind='sph'):
        cs_0 = GEOCSpherical(coords={'lat': lat_0, 'lon': lon_0, 'height': height_0})
        phi_0 = cs_0['phi']
        theta_0 = cs_0['theta']
        cs_0 = cs_0.to_cartesian()
        x_0 = cs_0['x']
        y_0 = cs_0['y']
        z_0 = cs_0['z']

        x = self.coords.x
        y = self.coords.y
        z = self.coords.z
        shape_x = x.shape

        v = np.array([x.flatten(), y.flatten(), z.flatten()]).T
        v_0 = np.array([x_0, y_0, z_0])
        v_0 = np.tile(v_0, (x.size, 1))

        R_1 = np.array([
            [-np.sin(phi_0), -np.cos(phi_0), 0],
            [np.cos(phi_0), -np.sin(phi_0), 0],
            [0, 0, 1]
        ])

        R_2 = np.array([
            [1, 0, 0],
            [0, np.cos(theta_0), -np.sin(theta_0)],
            [0, np.sin(theta_0), np.cos(theta_0)]
        ])

        v_new = (v - v_0) @ R_1 @ R_2

        x_new = np.reshape(v_new[:, 0], shape_x)
        y_new = np.reshape(v_new[:, 1], shape_x)
        z_new = np.reshape(v_new[:, 2], shape_x)
        cs_new = LENUCartesian(coords={'x': x_new, 'y': y_new, 'z': z_new}, lat_0=lat_0, lon_0=lon_0)

        if kind == 'sph':
            cs_new = cs_new.to_spherical()
        return cs_new
