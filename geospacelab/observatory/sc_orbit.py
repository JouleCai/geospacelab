import numpy as np
import netCDF4 as nc
import datetime
import pathlib
import cftime
import pickle

from sscws.sscws import SscWs
from sscws.coordinates import CoordinateComponent, CoordinateSystem, \
    SurfaceGeographicCoordinates

import geospacelab.cs as gsl_cs
from geospacelab import preferences as pfr
import geospacelab.toolbox.utilities.pydatetime as dttool
import geospacelab.toolbox.utilities.pylogging as mylog
from geospacelab.datahub.__dataset_base__ import DatasetSourced

default_dataset_attrs = {
    'data_root_dir': pfr.datahub_data_root_dir / 'SSCWS',
}

default_variable_names = [
                'SC_GEO_LAT', 'SC_GEO_LON', 'SC_GEO_ALT',
                'SC_GEO_X', 'SC_GEO_Y', 'SC_GEO_Z',
                'SC_GSE_X', 'SC_GSE_Y', 'SC_GSE_Z',
                'SC_AACGM_LAT', 'SC_AACGM_LON', 'SC_AACGM_MLT',
                'SC_DATETIME'
            ]


class OrbitPosition_SSCWS(DatasetSourced):

    def __init__(
            self,
            dt_fr: datetime.datetime = None,
            dt_to: datetime.datetime = None,
            sat_id: str = None,
            to_AACGM: bool = True,
            to_netcdf: bool = True,
            time_clip: bool = True,
            allow_download: bool = True,
            force_download: bool = False,
            **kwargs
    ):
        kwargs.update(**default_dataset_attrs)
        super().__init__(**kwargs)

        self.ssc = SscWs()
        self.dt_fr = dt_fr
        self.dt_to = dt_to
        self.sat_id = sat_id
        self.to_AACGM = to_AACGM
        self.to_netcdf = to_netcdf
        self.time_clip = time_clip
        self.force_download = force_download
        self.allow_download = allow_download
        self.load_data()

    def load_data(self, **kwargs):
        self.check_data_files(load_mode='AUTO', **kwargs)
        self._set_default_variables(
            default_variable_names,
            configured_variables={}
        )

        for file_path in self.data_file_paths:
            fnc = nc.Dataset(file_path)
            variables = {}
            variable_names = [
                'SC_GEO_LAT', 'SC_GEO_LON', 'SC_GEO_ALT',
                'SC_GEO_X', 'SC_GEO_Y', 'SC_GEO_Z',
                'SC_GSE_X', 'SC_GSE_Y', 'SC_GSE_Z',
                'SC_AACGM_LAT', 'SC_AACGM_LON', 'SC_AACGM_MLT',
                'UNIX_TIME'
            ]
            for var_name in variable_names:
                variables[var_name] = np.array(fnc[var_name]).reshape((fnc[var_name].shape[0], 1))

            time_units = fnc['UNIX_TIME'].units

            variables['SC_DATETIME'] = cftime.num2date(variables['UNIX_TIME'].flatten(),
                                                    units=time_units,
                                                    only_use_cftime_datetimes=False,
                                                    only_use_python_datetimes=True)

            variables['SC_DATETIME'] = np.reshape(variables['SC_DATETIME'], (fnc['UNIX_TIME'].shape[0], 1))
            fnc.close()

            for var_name in self._variables.keys():
                value = variables[var_name]
                self._variables[var_name].join(value)

            # self.select_beams(field_aligned=True)
        if self.time_clip:
            self.time_filter_by_range(var_datetime_name='SC_DATETIME')

    def search_data_files(self, **kwargs):

        dt_fr = self.dt_fr
        dt_to = self.dt_to

        diff_months = dttool.get_diff_months(dt_fr, dt_to)

        for nm in range(diff_months + 1):

            thismonth = dttool.get_next_n_months(self.dt_fr, nm)

            initial_file_dir = kwargs.pop(
                'initial_file_dir', self.data_root_dir / self.sat_id.upper()
            )

            file_patterns = [
                self.sat_id.upper(),
                thismonth.strftime('%Y%m'),
            ]
            # remove empty str
            file_patterns = [pattern for pattern in file_patterns if str(pattern)]
            search_pattern = '*' + '*'.join(file_patterns) + '*'

            done = super().search_data_files(
                initial_file_dir=initial_file_dir,
                search_pattern=search_pattern,
                allow_multiple_files=False,
            )
            # Validate file paths

            if (not done and self.allow_download) or self.force_download:
                done = self.download_data()
                if done:
                    done = super().search_data_files(
                        initial_file_dir=initial_file_dir,
                        search_pattern=search_pattern,
                        allow_multiple_files=True
                    )

        return done

    def download_data(self):
        done = False
        diff_months = (self.dt_to.year - self.dt_fr.year) * 12 + (self.dt_to.month - self.dt_fr.month) % 12
        mylog.simpleinfo.info("Searching the orbit data from NASA/SscWs ...")
        for nm in range(diff_months + 1):
            thismonth = dttool.get_next_n_months(self.dt_fr, nm)
            thismonthend = dttool.get_last_day_of_month(thismonth, end=True)
            dt_fr_str = thismonth.strftime('%Y-%m-%dT%H:%M:%SZ')
            dt_to_str = thismonthend.strftime('%Y-%m-%dT%H:%M:%SZ')

            result = self.ssc.get_locations(
                [self.sat_id],
                [dt_fr_str, dt_to_str], [CoordinateSystem.GEO, CoordinateSystem.GSE]
            )

            if not list(result['Data']):
                return None

            data = result['Data'][0]
            coords = data['Coordinates'][0]
            coords_gse = data['Coordinates'][1]
            dts = data['Time']

            coords_in = {'x': coords['X'] / 6371.2, 'y': coords['Y'] / 6371.2, 'z': coords['Z'] / 6371.2}
            cs_car = gsl_cs.GEOCCartesian(coords=coords_in, ut=dts)
            cs_sph = cs_car.to_spherical()
            orbits = {
                'SC_GEO_LAT': cs_sph['lat'],
                'SC_GEO_LON': cs_sph['lon'],
                'SC_GEO_ALT': cs_sph['height'],
                'SC_GEO_X': coords['X'],
                'SC_GEO_Y': coords['Y'],
                'SC_GEO_Z': coords['Z'],
                'SC_GSE_X': coords_gse['X'],
                'SC_GSE_Y': coords_gse['Y'],
                'SC_GSE_Z': coords_gse['Z'],
                'SC_DATETIME': dts,
            }
            if self.to_AACGM:
                cs_aacgm = cs_sph.to_AACGM(append_mlt=True)
                orbits.update(
                    **{
                        'SC_AACGM_LAT': cs_aacgm['lat'],
                        'SC_AACGM_LON': cs_aacgm['lon'],
                        'SC_AACGM_MLT': cs_aacgm['mlt'],
                    }
                )
            if self.to_netcdf:
                self.save_to_netcdf(orbits, dt_fr=thismonth, dt_to=thismonthend)
            done = True
        return done

    def save_to_netcdf(self, orbits, dt_fr=None, dt_to=None):
        sat_info = self.get_sat_info(self.sat_id)
        fp = self.data_root_dir / sat_info['Id'].upper() / \
             f"SSCWS_orbits_{sat_info['Id'].upper()}_{dt_fr.strftime('%Y%m')}_{str(sat_info['Resolution']) + 's'}.nc"
        fp.parent.resolve().mkdir(parents=True, exist_ok=True)
        fnc = nc.Dataset(fp, 'w')
        fnc.createDimension('UNIX_TIME', orbits['SC_DATETIME'].shape[0])

        fnc.title = f"{sat_info['Name']} orbits from {dt_fr.strftime('%Y%m%dT%H%M%S')} to {dt_to.strftime('%Y%m%dT%H%M%S')}"
        time = fnc.createVariable('UNIX_TIME', np.float64, ('UNIX_TIME',))
        time.units = 'seconds since 1970-01-01 00:00:00.0'
        geo_lat = fnc.createVariable('SC_GEO_LAT', np.float32, ('UNIX_TIME',))
        geo_lon = fnc.createVariable('SC_GEO_LON', np.float32, ('UNIX_TIME',))
        geo_alt = fnc.createVariable('SC_GEO_ALT', np.float32, ('UNIX_TIME',))
        geo_x = fnc.createVariable('SC_GEO_X', np.float32, ('UNIX_TIME',))
        geo_y = fnc.createVariable('SC_GEO_Y', np.float32, ('UNIX_TIME',))
        geo_z = fnc.createVariable('SC_GEO_Z', np.float32, ('UNIX_TIME',))
        gse_x = fnc.createVariable('SC_GSE_X', np.float32, ('UNIX_TIME',))
        gse_y = fnc.createVariable('SC_GSE_Y', np.float32, ('UNIX_TIME',))
        gse_z = fnc.createVariable('SC_GSE_Z', np.float32, ('UNIX_TIME',))

        time_array = np.array(
            cftime.date2num(orbits['SC_DATETIME'].flatten(), units='seconds since 1970-01-01 00:00:00.0'))
        time[::] = time_array[::]
        geo_lat[::] = orbits['SC_GEO_LAT'][::]
        geo_lon[::] = orbits['SC_GEO_LON'][::]
        geo_alt[::] = orbits['SC_GEO_ALT'][::]
        geo_x[::] = orbits['SC_GEO_X'][::]
        geo_y[::] = orbits['SC_GEO_Y'][::]
        geo_z[::] = orbits['SC_GEO_Z'][::]
        gse_x[::] = orbits['SC_GSE_X'][::]
        gse_y[::] = orbits['SC_GSE_Y'][::]
        gse_z[::] = orbits['SC_GSE_Z'][::]

        if self.to_AACGM:
            aa_lat = fnc.createVariable('SC_AACGM_LAT', np.float32, ('UNIX_TIME',))
            aa_lon = fnc.createVariable('SC_AACGM_LON', np.float32, ('UNIX_TIME',))
            aa_mlt = fnc.createVariable('SC_AACGM_MLT', np.float32, ('UNIX_TIME',))
            aa_lat[::] = orbits['SC_AACGM_LAT'][::]
            aa_lon[::] = orbits['SC_AACGM_LON'][::]
            aa_mlt[::] = orbits['SC_AACGM_MLT'][::]

        mylog.simpleinfo.info(
            f"The {sat_info['Name']} orbit data from" +
            f" {dt_fr.strftime('%Y%m%dT%H%M%S')} to {dt_to.strftime('%Y%m%dT%H%M%S')}" +
            f" has been saved to {fp}."
        )
        fnc.close()

    @staticmethod
    def list_satellites(reload=False, logging=True):
        file_path = pfr.datahub_data_root_dir / 'SSCWS' / 'SSCWS_info_satellites.pkl'
        if not file_path.is_file():
            file_path.parent.resolve().mkdir(exist_ok=True)
            reload = True

        if reload:
            ssc = SscWs()
            res = ssc.get_observatories()
            sat_list = res['Observatory']
            with open(file_path, 'wb') as fobj:
                pickle.dump(sat_list, fobj, pickle.HIGHEST_PROTOCOL)

        with open(file_path, 'rb') as fobj:
            sat_list = pickle.load(fobj)

        if logging:
            mylog.simpleinfo.info(
                '{:<20s}{:<30s}{:<20s}{:<25s}{:<25s}{:80s}'.format(
                    'ID', 'NAME', 'RESOLUTION (s)', 'FROM', 'TO', 'RESOURCE'
                )
            )
            for sat_info in sat_list:
                id = sat_info['Id']
                name = sat_info['Name']
                res = sat_info['Resolution']
                dt_fr = sat_info['StartTime']
                dt_to = sat_info['EndTime']
                resource = sat_info['ResourceId']

                mylog.simpleinfo.info(
                    '{:<20s}{:30s}{:<20d}{:<25s}{:<25s}{:50s}'.format(
                        id, name, res, dt_fr.strftime('%Y-%m-%d'), dt_to.strftime('%Y-%m-%d'), str(resource)
                    )
                )
        return sat_list

    @staticmethod
    def get_sat_info(sat_id):
        sat_list = OrbitPosition_SSCWS.list_satellites(logging=False)
        sat_ids = [sat_info['Id'] for sat_info in sat_list]
        try:
            ind = sat_ids.index(sat_id)
        except ValueError:
            raise KeyError('The satellite ID does not exist!')

        return sat_list[ind]

    @staticmethod
    def list_stations(reload=False, logging=True):
        file_path = pfr.datahub_data_root_dir / 'SSCWS' / 'SSCWS_info_ground_stations.pkl'
        if not file_path.is_file():
            file_path.parent.resolve().mkdir(exist_ok=True)
            reload = True

        if reload:
            ssc = SscWs()
            res = ssc.get_ground_stations()
            station_list = res['GroundStation']
            with open(file_path, 'wb') as fobj:
                pickle.dump(station_list, fobj, pickle.HIGHEST_PROTOCOL)

        with open(file_path, 'rb') as fobj:
            station_list = pickle.load(fobj)

        if logging:
            mylog.simpleinfo.info(
                '{:^20s}{:30s}{:30s}{:30s}'.format(
                    'ID', 'NAME', 'LATITUDE', 'LONGITUDE'
                )
            )
            for station_info in station_list:
                id = station_info['Id']
                name = station_info['Name']
                lat = station_info['Location']['Latitude']
                lon = station_info['Location']['Longitude']

                mylog.simpleinfo.info(
                    '{:^20s}{:30s}{:<30.2f}{:<30.2f}'.format(
                        id, name, lat, lon 
                    )
                )
        return station_list

    @staticmethod
    def get_station_info(station_id):
        station_list = OrbitPosition_SSCWS.list_satellites(logging=False)
        station_ids = [sat_info['Id'] for sat_info in station_list]
        try:
            ind = station_ids.index(station_id)
        except ValueError:
            raise KeyError('The satellite ID does not exist!')

        return station_list[ind]

    @staticmethod
    def calc_orbit_az_el(lat, lon, height):
        cs_geo = gsl_cs.GEOCSpherical(coords={'lat': lat, 'lon': lon, 'height':height})
        
        cs_geo = cs_geo.to_cartesian()
        
        
        

if __name__ == "__main__":
    dt_fr = datetime.datetime(2012, 1, 3,)
    dt_to = datetime.datetime(2012, 3, 3)
    orbit_info = OrbitPosition_SSCWS(dt_fr, dt_to, sat_id='dmspf16')