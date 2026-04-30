# Licensed under the BSD 3-Clause License
# Copyright (C) 2021 GeospaceLab (geospacelab)
# Author: Lei Cai, Space Physics and Astronomy, University of Oulu

import numpy as np
import datetime
import copy
import pathlib

import geospacelab.datahub as datahub
from geospacelab.datahub import DatabaseModel, FacilityModel, InstrumentModel, ProductModel
from geospacelab.datahub.sources.esa_eo import esaeo_database
from geospacelab.datahub.sources.esa_eo.swarm import swarm_facility
from geospacelab.config import prf
import geospacelab.toolbox.utilities.pybasic as basic
import geospacelab.toolbox.utilities.pylogging as mylog
import geospacelab.toolbox.utilities.pydatetime as dttool



# default_data_search_recursive = True

default_attrs_required = []


class Dataset(datahub.DatasetSourced):
    _default_variable_names = None
    _default_dataset_attrs = None
    _default_downloader = None
    _default_loader = None
    _default_variable_config = None
    
    def __init__(self, **kwargs):
        # kwargs = basic.dict_set_default(kwargs, **Dataset._default_dataset_attrs)

        super().__init__(**kwargs)

        self.database = kwargs.pop('database', 'ESA/EarthOnline')
        self.facility = kwargs.pop('facility', 'SWARM')
        self.instrument = kwargs.pop('instrument', 'EFI-TII')
        self.data_file_ext = kwargs.pop('data_file_ext', '.cdf')
        self.product = kwargs.pop('product', 'TCT02')
        self.product_version = kwargs.pop('product_version', '')
        self.data_file_versions = None
        self.allow_download = kwargs.pop('allow_download', False)
        self.force_download = kwargs.pop('force_download', False)
        self.dry_run = kwargs.pop('dry_run', False)
        self.quality_control = kwargs.pop('quality_control', False)
        self.calib_control = kwargs.pop('calib_control', False)
        self.add_AACGM = kwargs.pop('add_AACGM', False) 
        self.add_APEX = kwargs.pop('add_APEX', False)
        self._data_root_dir_init = copy.deepcopy(self.data_root_dir)    # Record the initial root dir

        self.sat_id = kwargs.pop('sat_id', 'A')

        self.metadata = None

        allow_load = kwargs.pop('allow_load', False)

        # self.config(**kwargs)

        if self.loader is None:
            self.loader = self._default_loader

        if self.downloader is None:
            self.downloader = self._default_downloader

        self._validate_attrs()

        if allow_load:
            self.load_data()

    def _validate_attrs(self):
        for attr_name in default_attrs_required:
            attr = getattr(self, attr_name)
            if not list(attr):
                mylog.StreamLogger.warning("The parameter {} is required before loading data!".format(attr_name))
        if not self.allow_download:
            if self.product_version in ['latest', '', None]:
                raise ValueError("The product version must be specified when downloading is not allowed.")

    def label(self, **kwargs):
        label = super().label()
        return label

    def load_data(self, **kwargs):
        self.check_data_files(**kwargs)

        self._set_default_variables(
            self._default_variable_names,
            configured_variables=self._default_variable_config.configured_variables
        )
        for file_path, product_version in zip(self.data_file_paths, self.data_file_versions):
            load_obj = self.loader(file_path, file_type='cdf', product_version=product_version)

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
    
    def time_filter_by_range(self, **kwargs):
        kwargs.setdefault('var_datetime_name', 'SC_DATETIME')
        kwargs.update({'var_datetime_name': 'DATETIME'})
        super().time_filter_by_range(**kwargs)
        kwargs.update({'var_datetime_name': 'DATETIME_QUAL'})
        super().time_filter_by_range(**kwargs)

    def add_GEO_LST(self, var_name_datetime='SC_DATETIME', var_name_glon='SC_GEO_LON'):
        import geospacelab.observatory.earth.sun_position as sun_position
        lons = self[var_name_glon].flatten()
        uts = self[var_name_datetime].flatten()

        lsts = sun_position.convert_datetime_longitude_to_local_solar_time(
            dts=uts, lons=lons
        )
        name_prefix = var_name_glon.replace('GEO_LON', '')
        var = self.add_variable(var_name=name_prefix + 'GEO_LST')
        var.value = np.array(lsts)[:, np.newaxis]
        var.label = 'LST'
        var.unit = 'h'
        var.depends = self[var_name_glon].depends
        return var
            
    
    def convert_to_APEX(self, var_name_glat='SC_GEO_LAT', var_name_glon='SC_GEO_LON', var_name_gr='SC_GEO_r', var_name_datetime='SC_DATETIME'):
        import geospacelab.cs as gsl_cs
        
        glats = self[var_name_glat].flatten()
        glons = self[var_name_glon].flatten()
        grs = self[var_name_gr].flatten()

        coords_in = {
            'lat': glats,
            'lon': glons,
            'r': grs
        }
        
        name_prefix = var_name_glat.replace('GEO_LAT', '')
        
        dts = self[var_name_datetime].value.flatten()
        cs_sph = gsl_cs.GEOCSpherical(coords=coords_in, ut=dts)
        cs_apex = cs_sph.to_APEX(append_mlt=True)
        self.add_variable(name_prefix + 'APEX_LAT')
        self.add_variable(name_prefix + 'APEX_LON')
        self.add_variable(name_prefix + 'APEX_MLT')
        self[name_prefix + 'APEX_LAT'].value = cs_apex['lat'].reshape(self[var_name_datetime].value.shape)
        self[name_prefix + 'APEX_LON'].value = cs_apex['lon'].reshape(self[var_name_datetime].value.shape)
        self[name_prefix + 'APEX_MLT'].value = cs_apex['mlt'].reshape(self[var_name_datetime].value.shape)

    def convert_to_AACGM(self, var_name_glat='SC_GEO_LAT', var_name_glon='SC_GEO_LON', var_name_gr='SC_GEO_r', var_name_datetime='SC_DATETIME'):
        import geospacelab.cs as gsl_cs
        
        glats = self[var_name_glat].flatten()
        glons = self[var_name_glon].flatten()
        grs = self[var_name_gr].flatten()

        coords_in = {
            'lat': glats,
            'lon': glons,
            'r': grs
        }

        dts = self[var_name_datetime].value.flatten()
        cs_sph = gsl_cs.GEOCSpherical(coords=coords_in, ut=dts)
        cs_aacgm = cs_sph.to_AACGM(append_mlt=True)
        
        name_prefix = var_name_glat.replace('GEO_LAT', '')
        self.add_variable(name_prefix + 'AACGM_LAT')
        self.add_variable(name_prefix + 'AACGM_LON')
        self.add_variable(name_prefix + 'AACGM_MLT')
        self[name_prefix + 'AACGM_LAT'].value = cs_aacgm['lat'].reshape(self[var_name_datetime].value.shape)
        self[name_prefix + 'AACGM_LON'].value = cs_aacgm['lon'].reshape(self[var_name_datetime].value.shape)
        self[name_prefix + 'AACGM_MLT'].value = cs_aacgm['mlt'].reshape(self[var_name_datetime].value.shape)
        
    def time_filter_by_flag(self, flag_name, condition=None):
        
        flag_values = self[flag_name].flatten()
        inds = condition(flag_values) if callable(condition) else None
        self.time_filter_by_inds(inds)

    def search_data_files(self, file_patterns=None, file_pattern_daily=True, **kwargs):

        dt_fr = self.dt_fr
        dt_to = self.dt_to
        
        diff_days = dttool.get_diff_days(dt_fr, dt_to)

        dt0 = dttool.get_start_of_the_day(dt_fr)
        
        file_patterns_0 = file_patterns if file_patterns is not None else []
        done = [False] * (diff_days + 1)
        for i in range(diff_days + 1):
            if self.product_version in ['latest', '', None]:
                continue
            file_patterns = copy.deepcopy(file_patterns_0)
            this_day = dt0 + datetime.timedelta(days=i)

            initial_file_dir = kwargs.pop(
                'initial_file_dir', self.data_root_dir
            )
            initial_file_dir = initial_file_dir / self.product_version / 'Sat_{}'.format(self.sat_id) / this_day.strftime("%Y")
            if file_pattern_daily:
                file_patterns.append(this_day.strftime('%Y%m%d') + 'T')
            else:
                raise NotImplementedError("Only daily file pattern is supported for now.")
            # remove empty str
            file_patterns = [pattern for pattern in file_patterns if str(pattern)]
            search_pattern = '*' + '*'.join(file_patterns) + '*'

            done[i] = super().search_data_files(
                initial_file_dir=initial_file_dir,
                search_pattern=search_pattern,
                allow_multiple_files=True,
            )
            # Validate file paths

        if (not any(done) and self.allow_download) or self.force_download:
            if self.product_version in ['latest', '', None]:
                mylog.simpleinfo.info("Searching the latest version of the data product on the server...")
            download_obj = self.download_data()
            file_paths = download_obj.file_paths_local
            files_record = download_obj._files_record_remote
            self.data_file_versions = files_record['product_version']
            self.data_file_paths = [pathlib.Path(fp).with_suffix(self.data_file_ext) for fp in file_paths]

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
            product_version=self.product_version,
            force_download=self.force_download,
            dry_run=self.dry_run
        )
        
        return download_obj

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
