# Licensed under the BSD 3-Clause License
# Copyright (C) 2021 GeospaceLab (geospacelab)
# Author: Lei Cai, Space Physics and Astronomy, University of Oulu

import numpy as np
import datetime

import geospacelab.datahub as datahub
from geospacelab.datahub import DatabaseModel, FacilityModel, InstrumentModel, ProductModel
from geospacelab.datahub.sources.esa_eo import esaeo_database
from geospacelab.datahub.sources.esa_eo.swarm import swarm_facility
from geospacelab import preferences as prf
import geospacelab.toolbox.utilities.pybasic as basic
import geospacelab.toolbox.utilities.pylogging as mylog
import geospacelab.toolbox.utilities.pydatetime as dttool
from geospacelab.datahub.sources.esa_eo.swarm.advanced.efi_tct.loader import Loader as default_Loader
import geospacelab.datahub.sources.esa_eo.swarm.advanced.efi_tct.variable_config as var_config


default_dataset_attrs = {
    'database': esaeo_database,
    'facility': swarm_facility,
    'instrument': 'EFI-TII',
    'product': 'TCT02',
    'data_file_ext': '.cdf',
    'data_root_dir': prf.datahub_data_root_dir / 'ESA' / 'SWARM' / 'Advanced' / 'EFI-TCT',
    'allow_load': True,
    'allow_download': False,
    'data_search_recursive': False,
    'label_fields': ['database', 'facility', 'product'],
    'load_mode': 'assigned',
    'time_clip': True,
}

default_variable_names = [
    'DATETIME', 'DATETIME_1', 'DATETIME_2',
    'GRID_MLAT', 'GRID_MLT',
    'GRID_Jr'
    ]

# default_data_search_recursive = True

default_attrs_required = []


class Dataset(datahub.DatasetModel):
    def __init__(self, **kwargs):
        kwargs = basic.dict_set_default(kwargs, **default_dataset_attrs)

        super().__init__(**kwargs)

        self.database = kwargs.pop('database', 'ESA/EarthOnline')
        self.facility = kwargs.pop('facility', 'SWARM')
        self.instrument = kwargs.pop('instrument', 'EFI-TII')
        self.product = kwargs.pop('product', 'TCT02')
        self.allow_download = kwargs.pop('allow_download', False)

        self.sat_id = kwargs.pop('sat_id', 'A')

        self.metadata = None

        allow_load = kwargs.pop('allow_load', False)

        # self.config(**kwargs)

        if self.loader is None:
            self.loader = default_Loader

        if self.downloader is None:
            self.downloader = None

        self._validate_attrs()

        if allow_load:
            self.load_data()

    def _validate_attrs(self):
        for attr_name in default_attrs_required:
            attr = getattr(self, attr_name)
            if not list(attr):
                mylog.StreamLogger.warning("The parameter {} is required before loading data!".format(attr_name))

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
            load_obj = self.loader(file_path, file_type='cdf')

            for var_name in self._variables.keys():
                value = load_obj.variables[var_name]
                self._variables[var_name].join(value)

            # self.select_beams(field_aligned=True)
        if self.time_clip:
            self.time_filter_by_range(var_datetime_name = 'SC_DATETIME')

    def get_time_ind(self, ut):
        delta_sectime = [delta_t.total_seconds() for delta_t in (self['DATETIME'].value.flatten() - ut)]
        ind = np.where(np.abs(delta_sectime) == np.min(np.abs(delta_sectime)))[0][0]
        return ind

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
                initial_file_dir = self.data_root_dir / self.sat_id.lower() / thisday.strftime("%Y%m%d")
            file_patterns = [
                self.sat_id.upper(),
                self.product.upper(),
                thisday.strftime("%Y%m%d"),
            ]
            if self.orbit_id is not None:
                file_patterns.append(self.orbit_id)
            # remove empty str
            file_patterns = [pattern for pattern in file_patterns if str(pattern)]

            search_pattern = '*' + '*'.join(file_patterns) + '*'

            if self.orbit_id is not None:
                multiple_files = False
            else:
                fp_log = initial_file_dir / 'EDR-AUR.full.log'
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

            if not done and self.allow_download:
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
            orbit_id=self.orbit_id, sat_id=self.sat_id, file_type=self.product.lower(),
            data_file_root_dir=self.data_root_dir)
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
