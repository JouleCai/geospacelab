import datetime
import numpy as np

from geospacelab.cs._cs_base import SpaceCSBase, SphericalCoordinates, CartesianCoordinates
import geospacelab.toolbox.utilities.pylogging as mylog
import geospacelab.toolbox.utilities.pybasic as pybasic


class GEO(SpaceCSBase):
    def __init__(self, coords=None, ut=None, **kwargs):
        
        new_coords = ['lat', 'lon', 'height']
        super().__init__(name='GEO', coords=coords, ut=ut, kind='sph', new_coords=new_coords, **kwargs)

    def to_AACGM(self, append_mlt=False):
        cs_geoc = self.to_GEOC()
        cs_new = cs_geoc.to_AACGM(append_mlt=append_mlt)
        return cs_new

    def to_APEX(self, append_mlt=False):
        cs_geoc = self.to_GEOC()
        cs_new = cs_geoc.to_aacgm(append_mlt=append_mlt)

    def to_LENU(self, lat_0=None, lon_0=None, height_0=None):
        """
        Convert GEO to local LENU
        :param lat_0: geographic latitude
        :param lon_0: geographic longitude
        :param height_0: altiude from the sea level
        :return: LENU object.
        """

        cs_0 = GEO(coords={'lat': lat_0, 'lon': lon_0, 'height': height_0})
        cs_0 = cs_0.to_GEOC()
        cs_0.coords.convert_latlon_to_thetaphi()
        cs_0.coords.convert_h_to_r()

        cs_geoc = self.to_GEOC()
        cs_new = cs_geoc.to_LENU(lat_0=cs_0['lat'], lon_0=cs_0['lon'], height_0=cs_0['height'])

        return cs_new

    def to_GEOC(self, kind='sph'):
        from geospacelab.cs import geopack
        cs_new = GEOC(ut=self.ut, kind='sph')
        hgeod = self.coords.height
        if self['lat_unit'] == 'degree':
            factor = np.pi / 180.
        elif self['lat_unit'] == 'radians':
            factor = 1.
        else:
            raise AttributeError
        xmugeod = self['lat'] * factor
        r, theta = geopack.geodgeo(hgeod, xmugeod, 1)
        r = r / cs_new.coords._Re            # unit in Re
        phi = self['lon'] * factor
        cs_new['theta'] = theta
        cs_new['phi'] = phi
        cs_new['r'] = r

        cs_new.coords.convert_thetaphi_to_latlon()
        cs_new.coords.convert_r_to_h()
        if kind == 'car':
            cs_new = cs_new.coords.to_car()
        return cs_new


class GEOD(GEO):
    def __init__(self, coords=None, ut=None, kind='sph', **kwargs):
        super().__init__(name='GEOD', coords=coords, ut=ut, kind='sph', **kwargs)


class LENU(SpaceCSBase):
    def __init__(self, coords=None, lat_0=None, lon_0=None, height_0=None, ut=None, kind='sph', **kwargs):
        if kind == 'sph':
            new_coords = ['az', 'el', 'range']
        elif kind == 'car':
            new_coords = ['E', 'N', 'Up']
        else:
            raise NotImplemented
        self.lat_0 = lat_0
        self.lon_0 = lon_0
        self.height_0 = height_0
        super().__init__(name='GEO', coords=coords, ut=ut, kind=kind, new_coords=new_coords, **kwargs)

    def to_GEO(self):
        cs_new = self.to_GEOC()
        cs_new = cs_new.to_GEO()
        return cs_new

    def to_GEOC(self):
        cs_0 = GEOC(coords={'lat': self.lat_0, 'lon': self.lon_0, 'height': self.height_0})
        cs_0.coords.convert_latlon_to_thetaphi()
        cs_0.coords.convert_h_to_r()
        phi_0 = cs_0['phi']
        theta_0 = cs_0['theta']
        cs_0 = cs_0.coords.to_car()
        x_0 = cs_0['x']
        y_0 = cs_0['y']
        z_0 = cs_0['z']

        if self.kind == 'sph':
            if not hasattr(self.coords, 'phi'):
                self.az_el_range_to_sph()
            cs_v = self.coords.to_car()
        else:
            cs_v = self
        x = cs_v['x']
        y = cs_v['y']
        z = cs_v['z']
        shape_x = x.shape

        v = np.array([x.flatten(), y.flatten(), z.flatten()]).T
        v_0 = np.array([x_0, y_0, z_0])
        v_0 = np.tile(v_0, (x.size, 1))

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

        v_new = v @ R_1 @ R_2 + v_0

        x_new = np.reshape(v_new[:, 0], shape_x)
        y_new = np.reshape(v_new[:, 1], shape_x)
        z_new = np.reshape(v_new[:, 2], shape_x)
        cs_new = GEOC(coords={'x': x_new, 'y': y_new, 'z': z_new}, kind='car')
        return cs_new


class GEOC(SpaceCSBase):
    def __init__(self, coords=None, ut=None, kind='sph', **kwargs):
        if kind=='sph':
            new_coords = ['phi', 'theta', 'r']
        elif kind == 'car':
            new_coords = ['x', 'y', 'z']

        super().__init__(name='GEOC', coords=coords, ut=ut, kind=kind, new_coords=new_coords, **kwargs)

    def to_AACGM(self, append_mlt=False):
        import aacgmv2 as aacgm
        from geospacelab.cs._aacgm import AACGM
        method_code = 'G2A'

        cs_in = self

        if not hasattr(cs_in.coords, 'lat'):
            cs_in.coords.convert_thetaphi_to_latlon()
        if not hasattr(cs_in.coords, 'height'):
            cs_in.coords.convert_r_to_h()

        ut_type = type(cs_in.ut)
        if ut_type is list:
            uts = np.array(cs_in.ut)
        elif ut_type is np.ndarray:
            uts = cs_in.ut

        if ut_type is datetime.datetime:
            lat, lon, r = aacgm.convert_latlon_arr(in_lat=cs_in.coords.lat,
                                                   in_lon=cs_in.coords.lon, height=cs_in.coords.height,
                                                   dtime=cs_in.ut, method_code=method_code)
        else:
            if uts.shape[0] != cs_in.coords.lat.shape[0]:
                mylog.StreamLogger.error("Datetimes must have the same length as cs!")
                return
            lat = np.empty_like(cs_in.coords.lat)
            lon = np.empty_like(cs_in.coords.lon)
            r = np.empty_like(cs_in.coords.lat)
            for ind_dt, dt in enumerate(uts.flatten()):
                # print(ind_dt, dt, cs.lat[ind_dt, 0])
                lat[ind_dt], lon[ind_dt], r[ind_dt] = aacgm.convert_latlon_arr(in_lat=cs_in.coords.lat[ind_dt],
                                                                               in_lon=cs_in.coords.lon[ind_dt],
                                                                               height=cs_in.coords.height[ind_dt],
                                                                               dtime=dt, method_code=method_code)
        cs_new = AACGM(coords={'lat': lat, 'lon': lon, 'r': r, 'r_unit': 'R_E'}, ut=cs_in.ut)
        if append_mlt:
            if ut_type is datetime.datetime:
                mlt = aacgm.convert_mlt(lon, cs_in.ut)
            else:
                mlt = np.empty_like(cs_in.coords.lat)
                for ind_dt, dt in enumerate(cs_in.ut.flatten()):
                    mlt[ind_dt] = aacgm.convert_mlt(lon[ind_dt], dt)
            cs_new.coords.add_coord('mlt', unit='h')
            cs_new.coords.mlt = mlt
        return cs_new

    def to_APEX(self, append_mlt=False):
        import apexpy as apex
        from geospacelab.cs._apex import APEX

        cs_in = self

        if not hasattr(cs_in.coords, 'lat'):
            cs_in.coords.convert_thetaphi_to_latlon()
        if not hasattr(cs_in.coords, 'height'):
            cs_in.coords.convert_r_to_h()

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

    def to_GEO(self):
        from geospacelab.cs import geopack
        if self.kind == 'car':
            cs_in = self.coords.to_sph()
        else:
            cs_in = self
        cs_new = GEO(ut=cs_in.ut)

        if cs_in['r_unit'] == 'Re':
            r = cs_in['r'] * cs_new.coords._Re
        else:
            r = cs_in['r']

        theta = cs_in['theta']

        h, lat = geopack.geodgeo(r, theta, -1)
        lon = cs_in['phi']

        factor = 180. / np.pi

        cs_new['height'] = h
        cs_new['lat'] = lat * factor
        cs_new['lon'] = lon * factor
        return cs_new

    def to_LENU(self, lat_0=None, lon_0=None, height_0=None):
        cs_0 = GEOC(coords={'lat': lat_0, 'lon': lon_0, 'height': height_0})
        cs_0.coords.convert_latlon_to_thetaphi()
        cs_0.coords.convert_h_to_r()
        phi_0 = cs_0['phi']
        theta_0 = cs_0['theta']
        cs_0 = cs_0.coords.to_car()
        x_0 = cs_0['x']
        y_0 = cs_0['y']
        z_0 = cs_0['z']

        if self.kind == 'sph':
            if not hasattr(self.coords, 'phi'):
                self.coords.convert_latlon_to_thetaphi()
                self.coords.convert_h_to_r()
            cs_geoc = self.coords.to_car()
        else:
            cs_geoc = self
        x = cs_geoc['x']
        y = cs_geoc['y']
        z = cs_geoc['z']
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
        cs_new = LENU(coords={'x': x_new, 'y': y_new, 'z': z_new}, lat_0=lat_0, lon_0=lon_0, kind='car')
        cs_new.add_az_el_range()
        return cs_new