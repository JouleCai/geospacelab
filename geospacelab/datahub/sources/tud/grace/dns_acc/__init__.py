# Licensed under the BSD 3-Clause License
# Copyright (C) 2021 GeospaceLab (geospacelab)
# Author: Lei Cai, Space Physics and Astronomy, University of Oulu

import numpy as np
import datetime

import geospacelab.datahub as datahub
from geospacelab.datahub import DatabaseModel, FacilityModel, InstrumentModel, ProductModel
from geospacelab.datahub.sources.tud import tud_database
from geospacelab.datahub.sources.tud.grace import grace_facility
from geospacelab.config import prf
import geospacelab.toolbox.utilities.pybasic as basic
import geospacelab.toolbox.utilities.pylogging as mylog
import geospacelab.toolbox.utilities.pydatetime as dttool
from geospacelab.datahub.sources.tud.grace.dns_acc.loader import Loader as default_Loader
from geospacelab.datahub.sources.tud.grace.dns_acc.downloader import Downloader as default_Downloader
import geospacelab.datahub.sources.tud.grace.dns_acc.variable_config as var_config


default_dataset_attrs = {
    'database': tud_database,
    'facility': grace_facility,
    'instrument': 'ACC',
    'product': 'DNS-ACC',
    'data_file_ext': 'txt',
    'product_version': 'v02',
    'data_root_dir': prf.datahub_data_root_dir / 'TUD' / 'GRACE',
    'allow_load': True,
    'allow_download': True,
    'force_download': False,
    'data_search_recursive': False,
    'add_AACGM': False,
    'add_APEX': False,
    'label_fields': ['database', 'facility', 'instrument', 'product', 'product_version'],
    'load_mode': 'AUTO',
    'time_clip': True,
}

default_variable_names_v01 = [
    'SC_DATETIME',
    'SC_GEO_LAT',
    'SC_GEO_LON',
    'SC_GEO_ALT',
    'SC_ARG_LAT',
    'SC_GEO_LST',
    'rho_n',
    ]
default_variable_names_v02 = [
    'SC_DATETIME',
    'SC_GEO_LAT',
    'SC_GEO_LON',
    'SC_GEO_ALT',
    'SC_ARG_LAT',
    'SC_GEO_LST',
    'rho_n',
    'rho_n_MEAN',
    'FLAG',
    'FLAG_MEAN',
    ]

# default_data_search_recursive = True

default_attrs_required = ['sat_id']


class Dataset(datahub.DatasetSourced):
    def __init__(self, **kwargs):
        kwargs = basic.dict_set_default(kwargs, **default_dataset_attrs)

        super().__init__(**kwargs)

        self.database = kwargs.pop('database', 'TUD')
        self.facility = kwargs.pop('facility', 'GRACE')
        self.instrument = kwargs.pop('instrument', 'ACC')
        self.product = kwargs.pop('product', 'DNS-ACC')
        self.product_version = kwargs.pop('product_version', 'v01')
        self.local_latest_version = ''
        self.allow_download = kwargs.pop('allow_download', False)
        self.force_download = kwargs.pop('force_download', False)
        self.download_dry_run = kwargs.pop('download_dry_run', False)
        
        self.add_AACGM = kwargs.pop('add_AACGM', False) 
        self.add_APEX = kwargs.pop('add_APEX', False)
        self._data_root_dir = self.data_root_dir    # Record the initial root dir

        self.sat_id = kwargs.pop('sat_id', 'A')

        self.metadata = {}

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
            if not attr:
                mylog.StreamLogger.warning("The parameter {} is required before loading data!".format(attr_name))

        self.data_root_dir = self.data_root_dir / self.product.upper() / self.product_version

    def label(self, **kwargs):
        label = super().label()
        return label

    def load_data(self, **kwargs):
        self.check_data_files(**kwargs)
        if self.product_version == 'v01':
            default_variable_names = default_variable_names_v01
        else:
            default_variable_names = default_variable_names_v02 
        self._set_default_variables(
            default_variable_names,
            configured_variables=var_config.configured_variables
        )
        
        if self.product_version == 'v01':
            if self.visual == 'on':
                self['rho_n'].error = None
                self['rho_n'].visual.plot_config.style = '1P' 
                
        for file_path in self.data_file_paths:
            load_obj = self.loader(file_path, file_type='txt', version=self.product_version)

            for var_name in self._variables.keys():
                value = load_obj.variables[var_name]
                self._variables[var_name].join(value)

            # self.select_beams(field_aligned=True)
        if self.time_clip:
            self.time_filter_by_range(var_datetime_name='SC_DATETIME')

        if self.add_AACGM:
            self.convert_to_AACGM()

        if self.add_APEX:
            self.convert_to_APEX()

    def add_GEO_LST(self):
        lons = self['SC_GEO_LON'].flatten()
        uts = self['SC_DATETIME'].flatten()
        lsts = [ut + datetime.timedelta(seconds=int(lon/15.*3600)) for ut, lon in zip(uts, lons)]
        lsts = [lst.hour + lst.minute/60. + lst.second/3600. for lst in lsts]
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
            'height': self['SC_GEO_ALT'].value.flatten()
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
            'height': self['SC_GEO_ALT'].value.flatten()
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

    def interp_evenly(self, time_res=None, time_res_o=10, dt_fr=None, dt_to=None, masked=False):
        from scipy.interpolate import interp1d
        import geospacelab.toolbox.utilities.numpymath as nm
        if time_res is None:
            time_res = time_res_o
        ds_new = datahub.DatasetUser(dt_fr=self.dt_fr, dt_to=self.dt_to, visual=self.visual)
        ds_new.clone_variables(self)
        dts = ds_new['SC_DATETIME'].value.flatten()
        dt0 = dttool.get_start_of_the_day(dts[0])
        x_0 = np.array([(dt - dt0).total_seconds() for dt in dts])
        if dt_fr is None:
            dt_fr = dts[0] - datetime.timedelta(
                seconds=np.floor(((dts[0] - ds_new.dt_fr).total_seconds() / time_res)) * time_res
            )
        if dt_to is None:
            dt_to = dts[-1] + datetime.timedelta(
                seconds=np.floor(((ds_new.dt_to - dts[-1]).total_seconds() / time_res)) * time_res
            )
        sec_fr = (dt_fr - dt0).total_seconds()
        sec_to = (dt_to - dt0).total_seconds()
        x_1 = np.arange(sec_fr, sec_to + time_res / 2, time_res)

        f = interp1d(x_0, x_0, kind='nearest', bounds_error=False, fill_value=(x_0[0], x_0[-1]))
        pseudo_x = f(x_1)
        mask = np.abs(pseudo_x-x_1) > time_res_o/1.5

        dts_new = np.array([dt0 + datetime.timedelta(seconds=sec) for sec in x_1])
        ds_new['SC_DATETIME'].value = dts_new.reshape((dts_new.size, 1))

        period_var_dict = {'SC_GEO_LON': 360.,
                           'SC_AACGM_LON': 360.,
                           'SC_APEX_LON': 360.,
                           'SC_GEO_LST': 24.,
                           'SC_AACGM_MLT': 24.,
                           'SC_APEX_MLT': 24}

        for var_name in ds_new.keys():
            if var_name in ['SC_DATETIME']:
                continue
            if var_name in period_var_dict.keys():
                var = ds_new[var_name].value.flatten()
                var_new = nm.interp_period_data(x_0, var, x_1, period=period_var_dict[var_name], method='linear', bounds_error=False)
            else:
                method = 'linear' if 'FLAG' not in var_name else 'nearest'
                var = ds_new[var_name].value.flatten()
                f = interp1d(x_0, var, kind=method, bounds_error=False)
                var_new = f(x_1)
            if masked:
                var_new = np.ma.array(var_new, mask=mask, fill_value=np.nan)
            else:
                var_new[mask] = np.nan
            ds_new[var_name].value = var_new.reshape((dts_new.size, 1))
        return ds_new

    def search_data_files(self, **kwargs):

        dt_fr = self.dt_fr
        dt_to = self.dt_to

        diff_months = dttool.get_diff_months(dt_fr, dt_to)

        dt0 = dttool.get_first_day_of_month(self.dt_fr)

        for i in range(diff_months + 1):
            this_day = dttool.get_next_n_months(dt0, i)

            initial_file_dir = kwargs.pop(
                'initial_file_dir', self.data_root_dir
            )

            file_patterns = [
                'G' + self.sat_id.upper(),
                self.product.upper().replace('-', '_'),
                this_day.strftime('%Y_%m'),
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
                        allow_multiple_files=False
                    )
        self.data_file_paths = np.unique(self.data_file_paths)
        return done

    def download_data(self, dt_fr=None, dt_to=None):
        if dt_fr is None:
            dt_fr = self.dt_fr
        if dt_to is None:
            dt_to = self.dt_to
        download_obj = self.downloader(
            dt_fr, dt_to,
            sat_id=self.sat_id,
            product=self.product,
            version=self.product_version,
            force_download=self.force_download,
            dry_run=self.download_dry_run
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
