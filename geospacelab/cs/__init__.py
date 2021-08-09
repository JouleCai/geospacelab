import geospacelab.toolbox.utilities.pylogging as mylog
import geospacelab.toolbox.utilities.pyclass as pyclass


class SpaceCoordinateSystem(object):
    def __init__(self, name=None, coords=None, ut=None, sph_or_car=None, **kwargs):

        self.name = name
        self.ut = ut
        self.sph_or_car = sph_or_car
        if sph_or_car == 'sph':
            self.coords = SphericalCoordinates(cs=self.name)
        elif sph_or_car == 'car':
            self.coords = CartesianCoordinates(cs=self.name)

        self.coords.config(**coords)

    def transform(self, cs_to_class=None, trans_func=None, **kwargs):
        coords, sph_or_car = trans_func(**kwargs)
        cs_new = cs_to_class(coords=coords, ut=self.ut, sph_or_car)
        return cs_new


class GEO(SpaceCoordinateSystem):
    def __init__(self, coords=None, ut=None, **kwargs):
        super().__init__(name='GEO', coords=coords, ut=None, sph_or_car='sph', **kwargs)

    def transform(self, cs_to=None, **kwargs):
        if cs_to.lower() == 'aacgm':
            pass


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
                        mylog.StreamLogger.error("Datetimes must have the same length as cs!")
                        return
                    lat = np.empty_like(coords.lat)
                    lon = np.empty_like(coords.lon)
                    r = np.empty_like(coords.lat)
                    for ind_dt, dt in enumerate(self.dt.flatten()):
                        # print(ind_dt, dt, cs.lat[ind_dt, 0])
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


class SphericalCoordinates(object):
    def __init__(self, cs):
        self.cs = cs
        self.kind = 'sph'
        self.lat = None
        self.lat_unit = 'degree'
        self.lon = None
        self.lon_unit = 'degree'
        self.alt = None
        self.alt_unit = 'km'
        self.r = None
        self.r_unit = 'km'

    def __call__(self, **kwargs):
        self.config(**kwargs)

    def to_car(self):
        raise NotImplemented

    def config(self, logging=True, **kwargs):
        pyclass.set_object_attributes(self, append=False, logging=logging, **kwargs)

    def add_coord(self, name, unit):
        setattr(self, name, None)
        setattr(self, name + '_unit', unit)


class CartesianCoordinates(object):
    def __init__(self, cs):
        self.cs = cs
        self.kind = 'car'
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

    def config(self, logging=True, **kwargs):
        pyclass.set_object_attributes(self, append=False, logging=logging, **kwargs)

    def add_coord(self, name, unit):
        setattr(self, name, None)
        setattr(self, name + '_unit', unit)





