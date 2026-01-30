# Licensed under the BSD 3-Clause License
# Copyright (C) 2021 GeospaceLab (geospacelab)
# Author: Lei Cai, Space Physics and Astronomy, University of Oulu

import datetime

import numpy as np

import geospacelab.datahub as datahub
from geospacelab.datahub import DatabaseModel, FacilityModel, InstrumentModel, ProductModel
from geospacelab.datahub.sources.jhuapl import jhuapl_database
from geospacelab.config import prf
import geospacelab.toolbox.utilities.pydatetime as dttool
import geospacelab.toolbox.utilities.pybasic as basic
import geospacelab.toolbox.utilities.pylogging as mylog
from geospacelab.datahub.sources.cdaweb.dmsp.ssusi.sdr_disk.loader import Loader as default_Loader
from geospacelab.datahub.sources.cdaweb.dmsp.ssusi.sdr_disk.downloader import Downloader as default_Downloader
import geospacelab.datahub.sources.cdaweb.dmsp.ssusi.sdr_disk.variable_config as var_config


default_dataset_attrs = {
    'database': jhuapl_database,
    'facility': 'DMSP',
    'instrument': 'SSUSI',
    'product': 'SDR_DISK',
    'data_file_ext': 'nc',
    'data_root_dir': prf.datahub_data_root_dir / 'CDAWeb' / 'DMSP' / 'SSUSI' / 'SDR_DISK',
    'allow_load': True,
    'allow_download': True,
    'data_search_recursive': False,
    'label_fields': ['database', 'facility', 'instrument', 'product'],
    'time_clip': False,
}

default_variable_names = [
    'STARTING_TIME', 'STOPPING_TIME', 'DATETIME',
    'FILE_VERSION', 'DATA_PRODUCT_VERSION', 'SOFTWARE_VERSION_NUMBER', 'CALIBRATION_PERIOD_VERSION', 'EMISSION_SPECTRA',
    'SC_DATETIME', 'SC_ORBIT_ID', 'SC_GEO_LAT', 'SC_GEO_LON', 'SC_GEO_ALT',
    'DISK_GEO_LAT', 'DISK_GEO_LON', 'DISK_GEO_ALT', 'DISK_SZA', 'DISK_SAA',
    'ACROSS_PIXEL_SIZE', 'ALONG_PIXEL_SIZE', 'EFFECTIVE_LOOK_ANGLE', 'EXPOSURE', 'SAA_COUNT', 'DARK_COUNT_CORRECTION',
    'SCATTER_LIGHT_1216_CORRECTION', 'SCATTER_LIGHT_1304_CORRECTION', 'OVERLAP_1304_1356_CORRECTION', 'LONG_WAVE_SCATTER_CORRECTION',
    'RED_LEAK_CORRECTION', 'DQI',
    'DISK_COUNTS_1216', 'DISK_COUNTS_1304', 'DISK_COUNTS_1356', 'DISK_COUNTS_LBHS', 'DISK_COUNTS_LBHL',
    'DISK_R_1216', 'DISK_R_1304', 'DISK_R_1356', 'DISK_R_LBHS', 'DISK_R_LBHL',
    'DISK_R_RECT_1216', 'DISK_R_RECT_1304', 'DISK_R_RECT_1356', 'DISK_R_RECT_LBHS', 'DISK_R_RECT_LBHL',
    'DISK_R_1216_ERROR', 'DISK_R_1304_ERROR', 'DISK_R_1356_ERROR', 'DISK_R_LBHS_ERROR', 'DISK_R_LBHL_ERROR',
    'DISK_R_RECT_1216_ERROR', 'DISK_R_RECT_1304_ERROR', 'DISK_R_RECT_1356_ERROR', 'DISK_R_RECT_LBHS_ERROR', 'DISK_R_RECT_LBHL_ERROR',
    'DISK_DECOMP_1216_ERROR', 'DISK_DECOMP_1304_ERROR', 'DISK_DECOMP_1356_ERROR', 'DISK_DECOMP_LBHS_ERROR', 'DISK_DECOMP_LBHL_ERROR',
    'DISK_CALIB_1216_ERROR', 'DISK_CALIB_1304_ERROR', 'DISK_CALIB_1356_ERROR', 'DISK_CALIB_LBHS_ERROR', 'DISK_CALIB_LBHL_ERROR',
    'DQI_1216', 'DQI_1304', 'DQI_1356', 'DQI_LBHS', 'DQI_LBHL'
]

# default_data_search_recursive = True

default_attrs_required = ['sat_id', 'orbit_id', 'pp_type', 'pole']


class Dataset(datahub.DatasetSourced):
    def __init__(self, **kwargs):
        kwargs = basic.dict_set_default(kwargs, **default_dataset_attrs)

        super().__init__(**kwargs)

        self.database = kwargs.pop('database', 'JHUAPL')
        self.facility = kwargs.pop('facility', 'DMSP')
        self.instrument = kwargs.pop('instrument', 'SSUSI')
        self.product = kwargs.pop('product', 'SDR_DISK')
        self.allow_download = kwargs.pop('allow_download', True)
        self.force_download = kwargs.pop('force_download', False)
        self.download_dry_run = kwargs.pop('download_dry_run', False)

        self.sat_id = kwargs.pop('sat_id', '')
        self.orbit_id = kwargs.pop('orbit_id', None)
        self.pole = kwargs.pop('pole', '')
        self.pp_type = kwargs.pop('pp_type', '')

        self.metadata = None

        allow_load = kwargs.pop('allow_load', False)

        # self.config(**kwargs)

        if self.loader is None:
            self.loader = default_Loader

        if self.downloader is None:
            self.downloader = default_Downloader

        self._validate_attrs()

        if allow_load:
            self.load_data()

    def _validate_attrs(self):
        for attr_name in default_attrs_required:
            attr = getattr(self, attr_name)
            if not str(attr):
                mylog.StreamLogger.warning("The parameter {} is required before loading data!".format(attr_name))
            if attr_name == 'orbit_id':
                if attr is None or attr == '':
                    mylog.StreamLogger.warning("For a fast process, it's better to specify the orbit id.")

    def label(self, **kwargs):
        label = super().label()
        return label

    def load_data(self, **kwargs):
        self.check_data_files(**kwargs)

        self._set_default_variables(
            default_variable_names,
            configured_variables=var_config.configured_variables
        )
        for file_path in self.data_file_paths:
            load_obj = self.loader(file_path, file_type=self.product.lower(), pole=self.pole, pp_type=self.pp_type)

            for var_name in self._variables.keys():
                if var_name == 'EMISSION_SPECTRA':
                    self._variables[var_name].value = load_obj.variables[var_name]
                    continue
                if var_name in ['DATETIME', 'STARTING_TIME', 'STOPPING_TIME',
                                'FILE_VERSION', 'DATA_PRODUCT_VERSION', 'SOFTWARE_VERSION_NUMBER',
                                'CALIBRATION_PERIOD_VERSION']:
                    value = np.array([load_obj.variables[var_name]])[np.newaxis, :]
                else:
                    value = np.empty((1, ), dtype=object)
                    value[0] = load_obj.variables[var_name]
                    # value = np.array([[load_obj.variables[var_name]]], dtype=object)
                self._variables[var_name].join(value)

            # self.orbit_id = load_obj.metadata['ORBIT_ID']
            # self.select_beams(field_aligned=True)
        if self.time_clip:
            self.time_filter_by_range()

    def get_time_ind(self, ut, time_res=20*60, var_datetime_name='DATETIME', edge_cutoff=False, **kwargs):
        ind = super().get_time_ind(ut, time_res=time_res, var_datetime_name=var_datetime_name, edge_cutoff=edge_cutoff, **kwargs)
        return ind
    
    def regriddata(self, disk_data=None, disk_geo_lat=None, disk_geo_lon=None, *, across_res=20., interp_method='linear'):
        from scipy.interpolate import interp1d, griddata
        
        data_pts = disk_data.flatten()
        ind_valid = np.where(np.isfinite(data_pts))[0]
        xd = range(disk_data.shape[0])
        yd = range(disk_data.shape[1])

        # create the new grids
        ps_across = self['ACROSS_PIXEL_SIZE'].value[0]
        xx = range(disk_data.shape[0])
        yy = []
        for ind in range(disk_data.shape[1]-1):
            ps1 = ps_across[ind]
            ps2 = ps_across[ind+1]
            n_insert = int(np.round((ps1 + ps2) / 2 / across_res) - 1)
            yy.extend(np.linspace(ind, ind+1-0.001, n_insert + 2)[0: -1])
        yy.extend([ind+1])
        # grid_x, grid_y = np.meshgrid(xx, yy, indexing='ij')
        
        factor = np.pi / 180.
        sin_glat = np.sin(disk_geo_lat * factor)
        itpf_sin = interp1d(yd, sin_glat, kind='cubic', bounds_error=False, fill_value='extrapolate')
        sin_glat_i = itpf_sin(yy)
        sin_glat_i[sin_glat_i>1.] = 1.
        sin_glat_i[sin_glat_i<-1.] = -1. 
        
        cos_glat = np.cos(disk_geo_lat * factor)
        itpf_cos = interp1d(yd, cos_glat, kind='cubic', bounds_error=False, fill_value='extrapolate')
        cos_glat_i = itpf_cos(yy)
        cos_glat_i[cos_glat_i>1.] = 1.
        cos_glat_i[cos_glat_i<-1.] = -1.  
        
        rad = np.sign(sin_glat_i) * (np.pi / 2 - np.arcsin(cos_glat_i))
        grid_lat = rad / factor
        
        sin_glon = np.sin(disk_geo_lon * factor)
        itpf_sin = interp1d(yd, sin_glon, kind='cubic', bounds_error=False, fill_value='extrapolate')
        sin_glon_i = itpf_sin(yy)
        sin_glon_i[sin_glon_i>1.] = 1.
        sin_glon_i[sin_glon_i<-1.] = -1. 
        
        cos_glon = np.cos(disk_geo_lon * factor)
        itpf_cos = interp1d(yd, cos_glon, kind='cubic', bounds_error=False, fill_value='extrapolate')
        cos_glon_i = itpf_cos(yy)
        cos_glon_i[cos_glon_i>1.] = 1.
        cos_glon_i[cos_glon_i<-1.] = -1.  
        
        rad = np.sign(sin_glon_i) * (np.pi / 2 - np.arcsin(cos_glon_i))
        grid_lon = rad / factor 
        
        itpf_data = interp1d(yd, disk_data, kind='linear', bounds_error=False, fill_value='extrapolate')
        grid_data = itpf_data(yy)
        return grid_lat, grid_lon, grid_data
        

    def search_data_files(self, **kwargs):
        dt_fr = self.dt_fr
        if self.dt_to.hour > 22:
            dt_to = self.dt_to + datetime.timedelta(days=1)
        else:
            dt_to = self.dt_to
        diff_days = dttool.get_diff_days(dt_fr, dt_to)
        dt0 = dttool.get_start_of_the_day(dt_fr)
        for i in range(diff_days + 1):
            thisday = dt0 + datetime.timedelta(days=i)
            initial_file_dir = kwargs.pop('initial_file_dir', None)
            
            if initial_file_dir is None:
                initial_file_dir = self.data_root_dir / self.sat_id.upper() / str(thisday.year) /thisday.strftime("%Y%m%d")

            file_patterns = [
                'dmsp' + self.sat_id.lower(),
                'sdr-disk',
                thisday.strftime("%Y%j") + 'T',
            ]
            
            if self.orbit_id is not None:
                file_patterns.extend(['REV', self.orbit_id])

            # remove empty str
            file_patterns = [pattern for pattern in file_patterns if str(pattern)]

            search_pattern = '*' + '*'.join(file_patterns) + '*'

            if self.orbit_id is not None:
                multiple_files = False
            else:
                fp_log = initial_file_dir / (self.product.upper() + '.full.log')
                if not fp_log.is_file():
                    self.download_data(dt_fr=thisday, dt_to=thisday)
                multiple_files = True
            done = super().search_data_files(
                initial_file_dir=initial_file_dir,
                search_pattern=search_pattern,
                allow_multiple_files=multiple_files,
            )
            if done and self.orbit_id is not None:
                return True

            # Validate file paths

            if (not done and self.allow_download) or (self.force_download):
                done = self.download_data(dt_fr=thisday, dt_to=thisday)
                if done:
                    done = super().search_data_files(
                        initial_file_dir=initial_file_dir,
                        search_pattern=search_pattern,
                        allow_multiple_files=multiple_files
                    )

        return done

    def download_data(self, dt_fr=None, dt_to=None):
        if dt_fr is None:
            dt_fr = self.dt_fr
        if dt_to is None:
            dt_to = self.dt_to
        download_obj = self.downloader(
            dt_fr, dt_to,
            orbit_id=self.orbit_id, sat_id=self.sat_id,
            root_dir_local=self.data_root_dir,
            force_download=self.force_download,
            dry_run=self.download_dry_run,
        )
        return any(download_obj.done)

    @property
    def database(self):
        return self._database

    @database.setter
    def database(self, value):
        if isinstance(value, str):
            self._database = DatabaseModel(value)
        elif issubclass(value.__class__, DatabaseModel):
            self._database = value
        else:
            raise TypeError

    @property
    def product(self):
        return self._product

    @product.setter
    def product(self, value):
        if isinstance(value, str):
            self._product = ProductModel(value)
        elif issubclass(value.__class__, ProductModel):
            self._product = value
        else:
            raise TypeError

    @property
    def facility(self):
        return self._facility

    @facility.setter
    def facility(self, value):
        if isinstance(value, str):
            self._facility = FacilityModel(value)
        elif issubclass(value.__class__, FacilityModel):
            self._facility = value
        else:
            raise TypeError

    @property
    def instrument(self):
        return self._instrument

    @instrument.setter
    def instrument(self, value):
        if isinstance(value, str):
            self._instrument = InstrumentModel(value)
        elif issubclass(value.__class__, InstrumentModel):
            self._instrument = value
        else:
            raise TypeError
