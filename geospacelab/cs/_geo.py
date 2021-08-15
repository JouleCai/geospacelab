import datetime
import numpy as np

from geospacelab.cs._cs_base import SpaceCSBase, SphericalCoordinates, CartesianCoordinates, SpaceCartesianCS, SpaceSphericalCS
import geospacelab.toolbox.utilities.pylogging as mylog
import geospacelab.toolbox.utilities.pybasic as pybasic


class GEO(SpaceSphericalCS):
    def __init__(self, coords=None, ut=None, vector=None, **kwargs):
        
        kwargs.setdefault('new_coords', ['lat', 'lon', 'height'])
        super().__init__(name='GEO', coords=coords, vector=vector, ut=ut, **kwargs)

    def to_AACGM(self, append_mlt=False):
        cs_geoc = self.to_GEOC(kind='sph')
        cs_new = cs_geoc.to_AACGM(append_mlt=append_mlt)
        return cs_new

    def to_APEX(self, append_mlt=False):
        cs_geoc = self.to_GEOC(kind='sph')
        cs_new = cs_geoc.to_aacgm(append_mlt=append_mlt)
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
        cs_geoc = cs_0.to_GEOC(kind='sph')

        cs_geoc = self.to_GEOC(kind='car')
        cs_new = cs_geoc.to_LENU(lat_0=cs_0['lat'], lon_0=cs_0['lon'], height_0=cs_0['height'], kind=kind)

        return cs_new

    def to_GEOC(self, kind='sph', **kwargs):
        from geospacelab.cs import geopack
        hgeod = self.coords.height
        if self['lat_unit'] == 'deg':
            factor = np.pi / 180.
        elif self['lat_unit'] == 'rad':
            factor = 1.
        else:
            raise AttributeError
        xmugeod = self['lat'] * factor
        r, theta = geopack.geodgeo(hgeod, xmugeod, 1)
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
        kwargs.setdefault('new_coords', ['E', 'N', 'Up'])
        self.lat_0 = lat_0
        self.lon_0 = lon_0
        self.height_0 = height_0
        super().__init__(name='LENU', coords=coords, ut=ut, **kwargs)
        if self.E is not None and self.x is None:
            self.convert_ENU_to_xyz()
        elif self.E is None and self.x is not None:
            self.convert_xyz_to_ENU()

    def convert_ENU_to_xyz(self):
        x = self.coords.E / self.coords.Re
        y = self.coords.N / self.coords.Re
        z = self.coords.Up / self.coords.Re
        self.coords.add_coord(name='x', value=x, unit='Re')
        self.coords.add_coord(name='y', value=y, unit='Re')
        self.coords.add_coord(name='z', value=z, unit='Re')

    def convert_xyz_to_ENU(self):
        E = self.coords.x * self.coords.Re
        N = self.coords.y * self.coords.Re
        Up = self.coords.z * self.coords.Re
        self.coords.add_coord(name='E', value=E, unit='km')
        self.coords.add_coord(name='N', value=N, unit='km')
        self.coords.add_coord(name='Up', value=Up, unit='km')

    def to_GEOC(self, kind='sph'):

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
            [0,     np.cos(theta_0),    -np.sin(theta_0)],
            [0,     np.sin(theta_0),    np.cos(theta_0)]
        ])

        R_2 = np.array([
            [-np.sin(phi_0),    -np.cos(phi_0),     0],
            [np.cos(phi_0),     -np.sin(phi_0),     0],
            [0,                 0,                  1]
        ])

        x = self.x
        y = self.y
        z = self.z
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

        cs_new = GEOC(coords=coords, vector=vector, kind='car')

        if kind == 'sph':
            cs_new = cs_new.to_spherical()
        return cs_new

    def to_GEO(self):
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
        if self.phi is None and self.az is not None:
            self.convert_rangeazel_to_rphitheta()
        elif self.phi is not None and self.az is None:
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
        self.coords.add_coord(name='phi', value=phi, unit='rad')
        self.coords.add_coord(name='theta', value=theta, unit='rad')
        self.coords.add_coord(name='r', value=r, unit='Re')

    def convert_rphitheta_to_rangeazel(self):
        if self.coords.phi_unit == 'deg':
            factor = 1
        elif self.coords.az_unit == 'rad':
            factor = 180. / np.pi
        else:
            raise NotImplemented
        az = np.mod(90. - self.phi*factor, 360)
        el = 90. - self.theta * factor
        range = self.r * self.coords.Re

        self.coords.add_coord(name='az', value=az, unit='deg')
        self.coords.add_coord(name='el', value=el, unit='deg')
        self.coords.add_coord(name='range', value=range, unit='Re')

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
        kwargs.setdefault('new_coords', ['phi', 'theta', 'r'])
        super().__init__(name='GEOC', coords=coords, ut=ut, vector=vector, **kwargs)

        self.check_coords()

    def check_coords(self):
        if self.lat is not None and self.theta is None:
            self.convert_latlon_to_phitheta()
        elif self.lat is None and self.theta is not None:
            self.convert_phitheta_to_latlon()
        if self.height is None and self.r is not None:
            self.convert_r_to_height()
        elif self.height is not None and self.r is None:
            self.convert_height_to_r()

    def convert_latlon_to_thetaphi(self):
        if self.coords.lon_unit == 'rad':
            factor = 1.
        elif self.coords.phi_unit == 'deg':
            factor = np.pi / 180.
        else:
            raise NotImplementedError
        theta = 90. - self.lat * factor
        phi = self.phi * factor
        self.add_coord('phi', value=phi)
        self.add_coord('theta', value=theta)

    def convert_thetaphi_to_latlon(self):

        if self.coords.phi_unit == 'rad':
            factor = 180. / np.pi
        elif self.coords.phi_unit == 'deg':
            factor = 1.
        else:
            raise NotImplementedError
        lat = 90. - self.theta * factor
        lon = self.phi * factor
        self.add_coord('lat', value=lat)
        self.add_coord('lon', value=lon)

    def convert_height_to_r(self):
        r = (self.height + self.coords.Re) / self.coords.Re
        self.add_coord('r', value=r)
        return r

    def convert_r_to_height(self):
        if self.coords.r_unit == 'Re':
            factor = self.coords.Re
        elif self.coords.r_unit == 'km':
            factor = 1.
        else:
            raise NotImplementedError
        height = self.r * factor - self.coords.Re
        self.add_coord('height', value=height)
        return height

    def to_LENU(self, kind='sph'):
        cs_new = self.to_cartesian()
        return cs_new.to_LENU(kind=kind)

    def to_GEO(self, **kwargs):
        from geospacelab.cs import geopack

        r = self.r * self.coords.Re
        theta = self.theta

        h, lat = geopack.geodgeo(r, theta, -1)
        lon = self.phi

        factor = 180. / np.pi

        coords = {'lat': lat*factor, 'lon': lon*factor, 'height': h}
        vector = self.vector
        # if vector is not None:
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

        if ut_type is datetime.datetime:
            lat, lon, r = aacgm.convert_latlon_arr(in_lat=self.lat,
                                                   in_lon=self.lon, height=self.height,
                                                   dtime=self.ut, method_code=method_code)
        else:
            if uts.shape[0] != self.lat.shape[0]:
                mylog.StreamLogger.error("Datetimes must have the same length as cs!")
                return
            lat = np.empty_like(self.lat)
            lon = np.empty_like(self.lon)
            r = np.empty_like(self.lat)
            for ind_dt, dt in enumerate(uts.flatten()):
                # print(ind_dt, dt, cs.lat[ind_dt, 0])
                lat[ind_dt], lon[ind_dt], r[ind_dt] = aacgm.convert_latlon_arr(in_lat=self.lat[ind_dt],
                                                                               in_lon=self.lon[ind_dt],
                                                                               height=self.height[ind_dt],
                                                                               dtime=dt, method_code=method_code)
        cs_new = AACGM(coords={'lat': lat, 'lon': lon, 'r': r, 'r_unit': 'R_E'}, ut=self.ut)
        if append_mlt:
            if ut_type is datetime.datetime:
                mlt = aacgm.convert_mlt(lon, self.ut)
            else:
                mlt = np.empty_like(self.coords.lat)
                for ind_dt, dt in enumerate(self.ut.flatten()):
                    mlt[ind_dt] = aacgm.convert_mlt(lon[ind_dt], dt)
            cs_new.coords.add_coord('mlt', unit='h')
            cs_new.coords.mlt = mlt
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
        if ut_type is datetime.datetime:
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

        super().__init__(name='GEOC', coords=coords, ut=ut, kind='car', vector=vector, **kwargs)

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

        x = self.x
        y = self.y
        z = self.z
        shape_x = x.shape

        v = np.array([x.flatten(), y.flatten(), z.flatten()]).T
        v_0 = np.array([x_0, y_0, z_0])
        v_0 = np.tile(v_0, (x.size, 1))

        R_1 = np.array([
            [-np.sin(phi_0), np.cos(phi_0), 0],
            [-np.cos(phi_0), -np.sin(phi_0), 0],
            [0, 0, 1]
        ])

        R_2 = np.array([
            [1, 0, 0],
            [0, np.cos(theta_0), np.sin(theta_0)],
            [0, -np.sin(theta_0), np.cos(theta_0)]
        ])

        v_new = (v - v_0) @ R_1 @ R_2

        x_new = np.reshape(v_new[:, 0], shape_x)
        y_new = np.reshape(v_new[:, 1], shape_x)
        z_new = np.reshape(v_new[:, 2], shape_x)
        cs_new = LENUCartesian(coords={'x': x_new, 'y': y_new, 'z': z_new}, lat_0=lat_0, lon_0=lon_0, kind='car')

        if kind == 'sph':
            cs_new = cs_new.to_spherical()
        return cs_new
