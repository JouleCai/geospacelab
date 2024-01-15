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
from geospacelab.datahub.sources.tud.grace.wnd_acc.loader import Loader as default_Loader
from geospacelab.datahub.sources.tud.grace.wnd_acc.downloader import Downloader as default_Downloader
import geospacelab.datahub.sources.tud.grace.wnd_acc.variable_config as var_config


default_dataset_attrs = {
    'database': tud_database,
    'facility': grace_facility,
    'instrument': 'ACC',
    'product': 'WND-ACC',
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

default_variable_names_v01 = []
default_variable_names_v02 = [
    'SC_DATETIME',
    'SC_GEO_LAT',
    'SC_GEO_LON',
    'SC_GEO_ALT',
    'SC_ARG_LAT',
    'SC_GEO_LST',
    'u_CROSS',
    'UNIT_VECTOR_N',
    'UNIT_VECTOR_E',
    'UNIT_VECTOR_D',
    'FLAG',
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
        self.product = kwargs.pop('product', 'WND-ACC')
        self.product_version = kwargs.pop('product_version', 'v01')
        self.local_latest_version = ''
        self.allow_download = kwargs.pop('allow_download', False)
        self.force_download = kwargs.pop('force_download', False)
        self.add_AACGM = kwargs.pop('add_AACGM', False) 
        self.add_APEX = kwargs.pop('add_APEX', False)
        self._data_root_dir = self.data_root_dir    # Record the initial root dir

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

        self._add_u_CT()

    def _add_u_CT(self):
        from geospacelab.observatory.orbit.utilities import LEOToolbox
        ds_leo = LEOToolbox(self.dt_fr, self.dt_to)
        ds_leo.clone_variables(self)

        wind_unit_vector = np.concatenate(
            (
                self['UNIT_VECTOR_N'].value,
                self['UNIT_VECTOR_E'].value,
                self['UNIT_VECTOR_D'].value
            ),
        axis=1
        )
        orbit_unit_vector = ds_leo.trajectory_local_unit_vector()
        cp = np.cross(orbit_unit_vector, wind_unit_vector)
        u_CT = -np.sign(cp[:, 2]) * self['u_CROSS'].value.flatten()

        var = self['u_CROSS'].clone()
        var.name = 'u_CT'
        var.label = r'$u_{CT}$'
        var.visual.axis[1].lim = [None, None]
        var.value = u_CT[:, np.newaxis]
        self['u_CT'] = var

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
                self.product_version + '.txt'
            ]
            # remove empty str
            file_patterns = [pattern for pattern in file_patterns if str(pattern)]
            search_pattern = '*' + '*'.join(file_patterns) + '*'

            done = super().search_data_files(
                initial_file_dir=initial_file_dir,
                search_pattern=search_pattern,
                allow_multiple_files=False,
                include_extension=False,
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
            force=self.force_download
        )

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
