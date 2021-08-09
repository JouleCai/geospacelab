import datetime
import numpy as np

from geospacelab.cs._cs_base import SpaceCoordinateSystem, SphericalCoordinates, CartesianCoordinates
import geospacelab.toolbox.utilities.pylogging as mylog


class GEO(SpaceCoordinateSystem):
    def __init__(self, coords=None, ut=None, **kwargs):
        super().__init__(name='GEO', coords=coords, ut=ut, sph_or_car='sph', **kwargs)

    def to_aacgm(self, append_mlt=False):
        import aacgmv2 as aacgm
        from geospacelab.cs._aacgm import AACGM
        method_code = 'G2A'

        ut_type = type(self.ut)
        if ut_type is list:
            uts = np.array(self.ut)
        elif ut_type is np.ndarray:
            uts = self.ut

        if ut_type is datetime.datetime:
            lat, lon, r = aacgm.convert_latlon_arr(in_lat=self.coords.lat,
                                                   in_lon=self.coords.lon, height=self.coords.h,
                                                   dtime=self.ut, method_code=method_code)
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
                                                                               height=self.coords.h[ind_dt],
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

    def to_apex(self, append_mlt=False):
        import apexpy as apex
        from geospacelab.cs._apex import APEX

        ut_type = type(self.ut)
        if ut_type is list:
            uts = np.array(self.ut)
        elif ut_type is np.ndarray:
            uts = self.ut

        mlt = None
        if ut_type is datetime.datetime:
            apex_obj = apex.Apex(self.ut)
            mlat, mlon = apex_obj.convert(
                self.coords.lat, self.coords.lon, 'geo', 'apex', height=self.coords.h,
            )
            if append_mlt:
                mlt = apex_obj.mlon2mlt(mlon, self.ut)
        else:
            if uts.shape[0] != self.coords.lat.shape[0]:
                mylog.StreamLogger.error("Datetimes must have the same length as cs!")
                return

            mlat = np.empty_like(self.coords.lat)
            mlon = np.empty_like(self.coords.lat)
            mlt = np.empty_like(self.coords.lat)
            for ind_dt, dt in enumerate(uts.flatten()):
                # print(ind_dt, dt, cs.lat[ind_dt, 0])
                apex_obj = apex.Apex(dt)
                mlat[ind_dt], mlon[ind_dt] = apex_obj.convert(
                    self.coords.lat[ind_dt], self.coords.lon[ind_dt], 'geo', 'apex',
                    height=self.coords.h[ind_dt]
                )
                if append_mlt:
                    mlt[ind_dt] = apex_obj.mlon2mlt(mlon[ind_dt], dt)

        cs_new = APEX(coords={'lat': mlat, 'lon': mlon, 'h': self.coords.h, 'mlt': mlt}, ut=self.ut)

        return cs_new

