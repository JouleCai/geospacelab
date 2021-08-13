import datetime
import numpy as np

from geospacelab.cs._cs_base import SpaceCSBase, SphericalCoordinates, CartesianCoordinates
import geospacelab.toolbox.utilities.pylogging as mylog
import geospacelab.toolbox.utilities.pybasic as pybasic


class GEO(SpaceCSBase):
    def __init__(self, coords=None, ut=None, kind='sph', **kwargs):
        
        new_coords = ['lat', 'lon', 'height']
        super().__init__(name='GEO', coords=coords, ut=ut, kind='sph', new_coords=new_coords, **kwargs)

    def to_AACGM(self, append_mlt=False):
        cs_geoc = self.to_GEOC()
        cs_new = cs_geoc.to_AACGM(append_mlt=append_mlt)
        return cs_new

    def to_APEX(self, append_mlt=False):
        cs_geoc = self.to_GEOC()
        cs_new = cs_geoc.to_aacgm(append_mlt=append_mlt)

    def to_LENU(self, lat_0, lon_0, height_0):
        """
        Convert GEO to local LENU
        :param lat_0: geographic latitude
        :param lon_0: geographic longitude
        :param height_0: altiude from the sea level
        :return: LENU object.
        """
        pass

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

        if kind == 'car':
            cs_new = cs_new.coords.to_car()
        return cs_new


class GEOD(GEO):
    def __init__(self, coords=None, ut=None, kind='sph', **kwargs):
        super().__init__(name='GEOD', coords=coords, ut=ut, kind='sph', **kwargs)


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
