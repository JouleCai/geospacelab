import geospacelab.toolbox.utilities.pylogging as mylog


class SpaceCS(object):
    def __init__(self, *args, cs=None, ut=None, sph_or_car=None, **kwargs):
        # args:     coordinates
        #
        nargs = len(args)
        if nargs == 1 and isinstance(args[0], tuple):
            coords = args[0]
        else:
            coords = args
        self.CS = kwargs.pop('CS', None)
        if self.CS is None:
            mylog.StreamLogger.error('The name of the CS must be declared!')
            return
        self.dt = kwargs.pop('dt', None)
        self.coords_type = kwargs.pop('coords_type', 'sph')
        self.coords_labels = kwargs.pop('coords_labels', None)
        if self.coords_labels is None:
            mylog.StreamLogger.error('coordinate labels must be declared!')
        self.coords_units = kwargs.pop('coords_units', None)
        self.coords = Coords(coords, self.coords_labels, self.coords_type, units=self.coords_units)

    def transform(self, CS_to=None, **kwargs):
        CS_fr = self.CS
        if 'AACGM' in [CS_fr, CS_to]:

            if CS_to == 'AACGM':
                if CS_fr == 'GEO':
                    coords = self.coords
                else:
                    csObj_temp = self.transform(CS_to='GEO')
                    coords = csObj_temp.coords
                method_code = 'G2A'
            else:
                method_code = 'A2G'
            if CS_fr != CS_to:
                if isinstance(self.dt, datetime.datetime):
                    lat, lon, r = aacgm.convert_latlon_arr(in_lat=coords.lat,
                                                           in_lon=coords.lon, height=coords.alt,
                                                           dtime=self.dt, method_code=method_code, )
                else:
                    if self.dt.shape[0] != coords.lat.shape[0]:
                        mylog.StreamLogger.error("Datetimes must have the same length as coords!")
                        return
                    lat = np.empty_like(coords.lat)
                    lon = np.empty_like(coords.lon)
                    r = np.empty_like(coords.lat)
                    for ind_dt, dt in enumerate(self.dt.flatten()):
                        # print(ind_dt, dt, coords.lat[ind_dt, 0])
                        lat[ind_dt], lon[ind_dt], r[ind_dt] = aacgm.convert_latlon_arr(in_lat=coords.lat[ind_dt],
                                                                                       in_lon=coords.lon[ind_dt], height=coords.alt[ind_dt],
                                                                                       dtime=dt, method_code=method_code)

                csObj_new = SpaceCS(lat, lon, r, CS=CS_to, coords_labels=['lat', 'lon', 'r'], dt=self.dt)
            if CS_to == 'AACGM' and kwargs.pop('MLT', False):
                if isinstance(self.dt, datetime.datetime):
                    mlt = aacgm.convert_mlt(lon, self.dt)
                else:
                    mlt = np.empty_like(coords.lat)
                    for ind_dt, dt in enumerate(self.dt.flatten()):
                        mlt[ind_dt] = aacgm.convert_mlt(lon[ind_dt], dt)
                csObj_new.coords.add_coord('mlt', mlt, type='sph', unit='h')

        return csObj_new


class Coordinates(object):
    def __init__(self, coords, labels, sph_or_car=None, units=None):
        self._standard_coords = {'sph': {'labels':   ['lon', 'lat', 'colat', 'alt', 'r', 'mlt'],
                                         'units':    ['degree', 'degree', 'degree', 'km', 'km', 'h']
                                         },
                                 'car': {'labels':   ['x', 'y', 'z'],
                                         'units':    ['km', 'km', 'km']
                                         }
                                 }
        if units is None:
            units = [None] * len(coords)
        for ind, c in enumerate(coords):
            label = labels[ind]
            if label not in self._standard_coords[sph_or_car]['labels']:
                mylog.StreamLogger.error('The label ' + label + ' is not a standard coordinate label!')
                return

            unit = units[ind]
            self.add_coord(label, c, sph_or_car=sph_or_car, unit=unit)

    def convert_car2sph(self):
        # from spacepy import irbempy as op
        # result = op.car2sph([self.x, self.y, self.z])
        pass

    def convert_sph2car(self):
        pass

    def convert_distance_unit(self, method='km2Re'):
        pass

    def add_coord(self, label, c, sph_or_car=None, unit=None):
        if unit is None:
            unit = self._standard_coords[sph_or_car]['units'][self._standard_coords[sph_or_car]['labels'].index(label)]
        setattr(self, label, c)
        setattr(self, label + '_unit', unit)





