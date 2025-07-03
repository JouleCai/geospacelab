# Licensed under the BSD 3-Clause License
# Copyright (C) 2021 GeospaceLab (geospacelab)
# Author: Lei Cai, Space Physics and Astronomy, University of Oulu

__author__ = "Lei Cai"
__copyright__ = "Copyright 2021, GeospaceLab"
__license__ = "BSD-3-Clause License"
__email__ = "lei.cai@oulu.fi"
__docformat__ = "reStructureText"

import numpy as np
import datetime

import geospacelab.datahub as datahub
from geospacelab.datahub import DatabaseModel, FacilityModel, InstrumentModel, ProductModel
from geospacelab.datahub.sources.madrigal import madrigal_database
from geospacelab.datahub.sources.madrigal.satellites.dmsp import dmsp_facility
from geospacelab.config import prf
import geospacelab.toolbox.utilities.pybasic as basic
import geospacelab.toolbox.utilities.pylogging as mylog
import geospacelab.toolbox.utilities.pydatetime as dttool
import geospacelab.cs as geo_cs
from geospacelab.datahub.sources.madrigal.satellites.dmsp.s1.loader import Loader as default_Loader
from geospacelab.datahub.sources.madrigal.satellites.dmsp.downloader import Downloader as default_Downloader
import geospacelab.datahub.sources.madrigal.satellites.dmsp.s1.variable_config as var_config


default_dataset_attrs = {
    'database': madrigal_database,
    'facility': dmsp_facility,
    'instrument': 'SSIES&SSM',
    'product': 's1',
    'data_file_ext': 'hdf5',
    'data_root_dir': prf.datahub_data_root_dir / 'Madrigal' / 'DMSP',
    'allow_load': True,
    'allow_download': True,
    'force_download': False,
    'data_search_recursive': False,
    'label_fields': ['database', 'facility', 'instrument', 'product'],
    'load_mode': 'AUTO',
    'time_clip': True,
    'add_AACGM': True,
    'add_APEX': False,
    'calib_orbit': True,
    'replace_orbit': True,
}

default_variable_names = [
    'SC_DATETIME',
    'SC_GEO_LAT',
    'SC_GEO_LON',
    'SC_GEO_ALT',
    'SC_MAG_LAT',
    'SC_MAG_LON',
    'SC_MAG_MLT',
    'n_e',
    'v_i_H',
    'v_i_V',
    'B_D',
    'B_F',
    'B_P',
    'd_B_D',
    'd_B_F',
    'd_B_P',
    ]

# default_data_search_recursive = True

default_attrs_required = ['sat_id', ]


class Dataset(datahub.DatasetSourced):
    def __init__(self, **kwargs):
        kwargs = basic.dict_set_default(kwargs, **default_dataset_attrs)

        super().__init__(**kwargs)

        self.database = kwargs.pop('database', 'Madrigal')
        self.facility = kwargs.pop('facility', 'DMSP')
        self.instrument = kwargs.pop('instrument', 'SSIES&SSM')
        self.product = kwargs.pop('product', 's1')
        self.allow_download = kwargs.pop('allow_download', False)
        self.force_download = kwargs.pop('force_download', False)
        self.add_AACGM = kwargs.pop('add_AACGM', False)
        self.add_APEX = kwargs.pop('add_APEX', False)
        self.calib_orbit = kwargs.pop('calib_orbit', False)
        self.replace_orbit = kwargs.pop('replace_orbit', False)

        self.sat_id = kwargs.pop('sat_id', None)

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
            load_obj = self.loader(file_path, file_ext='hdf5')

            for var_name in self._variables.keys():
                value = load_obj.variables[var_name]
                self._variables[var_name].join(value)

            # self.select_beams(field_aligned=True)
        if self.time_clip:
            self.time_filter_by_range(var_datetime_name='SC_DATETIME')

        if self.calib_orbit:
            self.fix_geo_lon()

        if self.add_AACGM:
            self.convert_to_AACGM()

        if self.add_APEX:
            self.convert_to_APEX()

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

    def fix_geo_lon(self):
        from geospacelab.observatory.orbit.sc_orbit import OrbitPosition_SSCWS
        from scipy.interpolate import CubicSpline

        dts_1 = self['SC_DATETIME'].value.flatten()
        dt0 = dttool.get_start_of_the_day(self.dt_fr)
        sectime_1 = [(dt - dt0).total_seconds() for dt in dts_1]
        glat_1 = self['SC_GEO_LAT'].flatten()
        glon_1 = self['SC_GEO_LON'].flatten()
        alt_1 = self['SC_GEO_ALT'].flatten()
        
        # check outliers
        orbit_obj = OrbitPosition_SSCWS(
            dt_fr=self.dt_fr - datetime.timedelta(minutes=30),
            dt_to=self.dt_to + datetime.timedelta(minutes=30),
            sat_id='dmsp' + self.sat_id.lower()
        )
        dts = orbit_obj['SC_DATETIME'].flatten()
        sectimes, _ = dttool.convert_datetime_to_sectime(dts, dt0=dt0)
        x = orbit_obj['SC_GEO_X'].flatten()
        y = orbit_obj['SC_GEO_Y'].flatten()
        z = orbit_obj['SC_GEO_Z'].flatten()
        f_x = CubicSpline(sectimes, x)
        x_i = f_x(sectime_1)
        f_y = CubicSpline(sectimes, y)
        y_i = f_y(sectime_1)
        f_z = CubicSpline(sectimes, z)
        z_i = f_z(sectime_1)
        
        
        cs = geo_cs.GEOCCartesian(
            coords={'x': x_i, 'y': y_i, 'z': z_i, 'x_unit': 'km', 'y_unit': 'km', 'z_unit': 'km'}
        )
        cs_new = cs.to_spherical()
        
        # glat_2 = cs_new['lat']
        glon_2 = cs_new['lon']
        # alt_2 = cs_new['height']
        
        delta = np.sin(glon_2 * np.pi / 180.) - np.sin(glon_1 * np.pi / 180.)
        
        if self.replace_orbit:
            glon_1 = glon_2
        else:
            glon_1[np.abs(delta)>0.001] = glon_2[np.abs(delta)>0.001]
            
        self['SC_GEO_LON'].value = glon_1[:, np.newaxis]

    def search_data_files(self, **kwargs):

        dt_fr = self.dt_fr
        dt_to = self.dt_to

        diff_days = dttool.get_diff_days(dt_fr, dt_to)

        dt0 = dttool.get_start_of_the_day(dt_fr)

        for i in range(diff_days + 1):
            this_day = dt0 + datetime.timedelta(days=i)

            initial_file_dir = kwargs.pop(
                'initial_file_dir', self.data_root_dir / this_day.strftime('%Y%m') / this_day.strftime('%Y%m%d')
            )

            file_patterns = [
                'dms',
                this_day.strftime('%Y%m%d'),
                self.sat_id[1:] + self.product,
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

    def download_data(self, dt_fr=None, dt_to=None):
        if dt_fr is None:
            dt_fr = self.dt_fr
        if dt_to is None:
            dt_to = self.dt_to
        download_obj = self.downloader(
            dt_fr, dt_to,
            sat_id=self.sat_id,
            file_type=self.product,
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
