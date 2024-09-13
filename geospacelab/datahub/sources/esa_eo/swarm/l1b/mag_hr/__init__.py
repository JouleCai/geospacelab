# Licensed under the BSD 3-Clause License
# Copyright (C) 2021 GeospaceLab (geospacelab)
# Author: Lei Cai, Space Physics and Astronomy, University of Oulu

import numpy as np
import datetime
import copy

import geospacelab.datahub as datahub
from geospacelab.datahub import DatabaseModel, FacilityModel, InstrumentModel, ProductModel
from geospacelab.datahub.sources.esa_eo import esaeo_database
from geospacelab.datahub.sources.esa_eo.swarm import swarm_facility
from geospacelab.config import prf
import geospacelab.toolbox.utilities.pybasic as basic
import geospacelab.toolbox.utilities.pylogging as mylog
import geospacelab.toolbox.utilities.pydatetime as dttool
from geospacelab.datahub.sources.esa_eo.swarm.l1b.mag_hr.loader import Loader as default_Loader
from geospacelab.datahub.sources.esa_eo.swarm.l1b.mag_hr.downloader import Downloader as default_Downloader
import geospacelab.datahub.sources.esa_eo.swarm.l1b.mag_hr.variable_config as var_config


default_dataset_attrs = {
    'database': esaeo_database,
    'facility': swarm_facility,
    'instrument': 'MAG',
    'product': 'HR_1B',
    'data_file_ext': 'cdf',
    'product_version': 'latest',
    'data_root_dir': prf.datahub_data_root_dir / 'ESA' / 'SWARM' / 'Level1b' / 'MAG_HR',
    'allow_load': True,
    'allow_download': True,
    'force_download': False,
    'data_search_recursive': False,
    'add_AACGM': False,
    'add_APEX': False,
    'quality_control': False,
    'calib_control': False,
    'label_fields': ['database', 'facility', 'instrument', 'product'],
    'load_mode': 'AUTO',
    'time_clip': True,
}

default_variable_names = [
    'SC_DATETIME',
    'SC_GEO_LAT',
    'SC_GEO_LON',
    'SC_GEO_r',
    'B_NEC',
    'B_N',
    'B_E',
    'B_C',
    'FLAG_B',
    'FLAG_q',
    'FLAG_Platform',
    ]

# default_data_search_recursive = True

default_attrs_required = []


class Dataset(datahub.DatasetSourced):
    def __init__(self, **kwargs):
        kwargs = basic.dict_set_default(kwargs, **default_dataset_attrs)

        super().__init__(**kwargs)

        self.database = kwargs.pop('database', 'ESA/EarthOnline')
        self.facility = kwargs.pop('facility', 'SWARM')
        self.instrument = kwargs.pop('instrument', 'MAG')
        self.product = kwargs.pop('product', 'MAG_HR')
        self.product_version = kwargs.pop('product_version', '')
        self.local_latest_version = ''
        self.allow_download = kwargs.pop('allow_download', False)
        self.force_download = kwargs.pop('force_download', False)
        self.quality_control = kwargs.pop('quality_control', False)
        self.calib_control = kwargs.pop('calib_control', False)
        self.add_AACGM = kwargs.pop('add_AACGM', False)
        self.add_APEX = kwargs.pop('add_APEX', False)
        self._data_root_dir_init = copy.deepcopy(self.data_root_dir)   # Record the initial root dir

        self.sat_id = kwargs.pop('sat_id', 'A')

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
            if not list(attr):
                mylog.StreamLogger.warning("The parameter {} is required before loading data!".format(attr_name))

        # self.data_root_dir = self.data_root_dir / self.product

        if str(self.product_version) and self.product_version != 'latest':
            self.data_root_dir = self.data_root_dir / self.product_version
        else:
            self.product_version = 'latest'
            self.force_download = False
            mylog.simpleinfo.info(f'Checking the latest version of the data file ...')
            self.download_data()
            
            # try:
            #     dirs_product_version = [f.name for f in self.data_root_dir.iterdir() if f.is_dir()]
            # except FileNotFoundError:
            #     dirs_product_version = []
            #     self.force_download = True
            # else:
            #     if not list(dirs_product_version):
            #         self.force_download = True

            # if list(dirs_product_version):
            #     self.local_latest_version = max(dirs_product_version)
            #     self.data_root_dir = self.data_root_dir / self.local_latest_version
            #     if not self.force_download:
            #         mylog.simpleinfo.info(
            #             "Note: Loading the local files " +
            #             "with the latest version {} ".format(self.local_latest_version) +
            #             "Keep an eye on the latest baselines online!"
            #         )

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
            load_obj = self.loader(file_path, file_type='cdf', dt_fr=self.dt_fr, dt_to=self.dt_to)

            for var_name in self._variables.keys():
                value = load_obj.variables[var_name]
                self._variables[var_name].join(value)

            # self.select_beams(field_aligned=True)
        if self.time_clip:
            self.time_filter_by_range(var_datetime_name='SC_DATETIME')
        if self.quality_control:
            self.time_filter_by_quality()
        if self.calib_control:
            self.time_filter_by_calib()
            
        if self.add_AACGM:
            self.convert_to_AACGM()

        if self.add_APEX:
            self.convert_to_APEX()

        self.add_GEO_LST()

    def add_GEO_LST(self):
        lons = self['SC_GEO_LON'].flatten()
        uts = self['SC_DATETIME'].flatten()
        lsts = [ut + datetime.timedelta(hours=lon / 15.) for ut, lon in zip(uts, lons)]
        lsts = [lst.hour + lst.minute / 60. + lst.second / 3600. for lst in lsts]
        var = self.add_variable(var_name='SC_GEO_LST')
        var.value = np.array(lsts)[:, np.newaxis]
        var.label = 'LST'
        var.unit = 'h'
        var.depends = self['SC_GEO_LON'].depends
        return var
    
    def convert_to_APEX(self):
        import geospacelab.cs as gsl_cs

        coords_in = {
            'lat': self['SC_GEO_LAT'].value.flatten(),
            'lon': self['SC_GEO_LON'].value.flatten(),
            'r': self['SC_GEO_r'].value.flatten() / 6371.2
        }
        dts = self['SC_DATETIME'].value.flatten()
        cs_sph = gsl_cs.GEOCSpherical(coords=coords_in, ut=dts)
        cs_apex = cs_sph.to_APEX(append_mlt=True)
        self.add_variable('SC_APEX_LAT')
        self.add_variable('SC_APEX_LON')
        self.add_variable('SC_APEX_MLT')
        self['SC_APEX_LAT'].value = cs_apex['lat'].reshape(self['SC_DATETIME'].value.shape)
        self['SC_APEX_LON'].value = cs_apex['lon'].reshape(self['SC_DATETIME'].value.shape)
        self['SC_APEX_MLT'].value = cs_apex['mlt'].reshape(self['SC_DATETIME'].value.shape)

    def convert_to_AACGM(self):
        import geospacelab.cs as gsl_cs

        coords_in = {
            'lat': self['SC_GEO_LAT'].value.flatten(),
            'lon': self['SC_GEO_LON'].value.flatten(),
            'r': self['SC_GEO_r'].value.flatten() / 6371.2
        }
        dts = self['SC_DATETIME'].value.flatten()
        cs_sph = gsl_cs.GEOCSpherical(coords=coords_in, ut=dts)
        cs_aacgm = cs_sph.to_AACGM(append_mlt=True)
        self.add_variable('SC_AACGM_LAT')
        self.add_variable('SC_AACGM_LON')
        self.add_variable('SC_AACGM_MLT')
        self['SC_AACGM_LAT'].value = cs_aacgm['lat'].reshape(self['SC_DATETIME'].value.shape)
        self['SC_AACGM_LON'].value = cs_aacgm['lon'].reshape(self['SC_DATETIME'].value.shape)
        self['SC_AACGM_MLT'].value = cs_aacgm['mlt'].reshape(self['SC_DATETIME'].value.shape)

    def time_filter_by_quality(self, quality_flags=None):
        if quality_flags is None:
            quality_flags = np.array([1])

        for qf in quality_flags:
            inds = np.where(self['QUALITY_FLAG'].value.flatten() == qf)[0]
            for key in self.keys():
                self._variables[key].value = self._variables[key].value[inds, ::]

    def time_filter_by_calib(self, calib_flags=None):

        if calib_flags is None:
            calib_flags = np.array([0])

        for cf in calib_flags:
            inds = np.where(self['CALIB_FLAG'].value.flatten() == cf)[0]
            for key in self.keys():
                self._variables[key].value = self._variables[key].value[inds, ::]

    def search_data_files(self, **kwargs):

        dt_fr = self.dt_fr
        dt_to = self.dt_to

        diff_days = dttool.get_diff_days(dt_fr, dt_to)

        dt0 = dttool.get_start_of_the_day(dt_fr)

        for i in range(diff_days + 1):
            this_day = dt0 + datetime.timedelta(days=i)

            initial_file_dir = kwargs.pop(
                'initial_file_dir', self.data_root_dir
            )
            initial_file_dir = initial_file_dir / 'Sat_{}'.format(self.sat_id) / this_day.strftime("%Y")

            file_patterns = [
                'MAG' + self.sat_id.upper(),
                self.product.upper(),
                this_day.strftime('%Y%m%d') + 'T',
                'MAG_HR'
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
                    initial_file_dir = self.data_root_dir
                    done = super().search_data_files(
                        initial_file_dir=initial_file_dir,
                        search_pattern=search_pattern,
                        allow_multiple_files=True
                    )

        return done

    def download_data(self, dt_fr=None, dt_to=None):
        if dt_fr is None:
            dt_fr = self.dt_fr
        if dt_to is None:
            dt_to = self.dt_to
        download_obj = self.downloader(
            dt_fr, dt_to,
            sat_id=self.sat_id,
            data_type=self.product,
            file_version=self.product_version,
            force=self.force_download
        )
        if download_obj.done:
            self.force_download = False

            if download_obj.file_version != self.local_latest_version and self.product_version == 'latest':
                mylog.simpleinfo.warning(
                    f"NOTE: The data with the latest version ({download_obj.file_version}) have been downloaded"
                )
            self.product_version = download_obj.file_version
            self.data_root_dir = copy.deepcopy(self._data_root_dir_init)
            self._validate_attrs()

        return download_obj.done

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
