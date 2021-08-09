import datetime
import numpy as np

from geospacelab.cs._cs_base import SpaceCoordinateSystem, SphericalCoordinates, CartesianCoordinates
import geospacelab.toolbox.utilities.pylogging as mylog


class GEO(SpaceCoordinateSystem):
    def __init__(self, coords=None, ut=None, **kwargs):
        super().__init__(name='GEO', coords=coords, ut=None, sph_or_car='sph', **kwargs)

    def to_AACGM(self):
        import aacgmv2 as aacgm
        method_code = 'G2A'

        ut_type = type(self.ut)
        if ut_type is list:
            uts = np.array(self.ut)
        elif ut_type is np.ndarray:
            uts = self.ut
        else:
            raise TypeError

        if ut_type is datetime.datetime:
            lat, lon, r = aacgm.convert_latlon_arr(in_lat=self.coords.lat,
                                                   in_lon=self.coords.lon, height=self.coords.alt,
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
                                                                               height=self.coords.alt[ind_dt],
                                                                               dtime=dt, method_code=method_code)

