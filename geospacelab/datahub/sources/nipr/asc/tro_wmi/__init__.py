# Licensed under the BSD 3-Clause License
# Copyright (C) 2021 GeospaceLab (geospacelab)
# Author: Lei Cai, Space Physics and Astronomy, University of Oulu

__author__ = "Lei Cai"
__copyright__ = "Copyright 2021, GeospaceLAB"
__license__ = "BSD-3-Clause License"
__email__ = "lei.cai@oulu.fi"
__docformat__ = "reStructureText"


import datetime
import numpy as np

import geospacelab.datahub as datahub
from geospacelab.datahub import DatabaseModel, FacilityModel, SiteModel
from geospacelab.datahub.sources.nipr import nipr_database
from geospacelab.config import prf
import geospacelab.toolbox.utilities.pydatetime as dttool
import geospacelab.toolbox.utilities.pybasic as basic
from geospacelab.datahub.sources.nipr.asc.tro_wmi.loader import Loader as default_Loader
import geospacelab.datahub.sources.nipr.asc.tro_wmi.variable_config as var_config
from geospacelab.datahub.sources.madrigal.isr.eiscat.utilities import *

default_dataset_attrs = {
    'kind': 'sourced',
    'database': nipr_database,
    'facility': 'NIPR-ASC',
    'data_file_type': 'image',
    'data_file_ext': 'jpg',
    'data_root_dir': prf.datahub_data_root_dir / 'NIPR' / 'ASC',
    'allow_download': False,
    'data_search_recursive': True,
    'beam_location': True,
    'label_fields': ['database', 'facility', 'site', 'antenna', 'experiment'],
}

default_variable_names = [
    'DATETIME',
    'ASC_IMG_DATA',
    'ASC_IMG_AZ',
    'ASC_IMG_EL'
]

# default_data_search_recursive = True

default_attrs_required = ['site', 'channel']


class Dataset(datahub.DatasetSourced):
    def __init__(self, **kwargs):
        kwargs = basic.dict_set_default(kwargs, **default_dataset_attrs)

        super().__init__(**kwargs)

        self.database = kwargs.pop('database', '')
        self.facility = kwargs.pop('facility', '')
        self.site = kwargs.pop('site', 'TRO')
        self.channel = kwargs.pop('channel', '')
        self.data_file_type = kwargs.pop('data_file_type', '')
        self.affiliation = kwargs.pop('affiliation', '')
        self.allow_download = kwargs.pop('allow_download', False)
        self.metadata = None

        allow_load = kwargs.pop('allow_load', False)

        # self.config(**kwargs)

        if self.loader is None:
            self.loader = default_Loader

        self._set_default_variables(
            default_variable_names,
            configured_variables=var_config.configured_variables
        )

        self._validate_attrs()

        if allow_load:
            self.load_data()

    def _validate_attrs(self):
        for attr_name in default_attrs_required:
            attr = getattr(self, attr_name)
            if not str(attr):
                mylog.StreamLogger.warning("The parameter {} is required before loading data!".format(attr_name))

        if str(self.data_file_type):
            self.data_file_ext = self.data_file_type.split('-')[1]

    def label(self, **kwargs):
        label = super().label()
        return label

    def load_data(self, **kwargs):
        self.check_data_files(**kwargs)

        for file_path in self.data_file_paths:
            load_obj = self.loader(file_path, file_type=self.data_file_type)

            for var_name in self._variables.keys():
                if var_name in ['ASC_IMG_AZ', 'ASC_IMG_EL']:
                    self._variables[var_name] = load_obj.variables[var_name]
                self._variables[var_name].join(load_obj.variables[var_name])

            self.metadata = load_obj.metadata
            # self.select_beams(field_aligned=True)
        if self.time_clip:
            self.time_filter_by_range()

    def calc_lat_lon(self, AACGM=True, APEX=False):
        from geospacelab.cs import LENUSpherical
        az = self['AZ'].value
        el = self['EL'].value
        range = self['RANGE'].value
        az = np.tile(az, (1, range.shape[1]))  # make az, el, range in the same shape
        el = np.tile(el, (1, range.shape[1]))

        # az = np.array([0, 90, 180, 270])
        # el = np.array([0, 45, 90, 45])
        # range = np.array([0, 100, 200, 100])
        cs_LENU = LENUSpherical(coords={'az': az, 'el': el, 'range': range},
                                lat_0=self.site.location['GEO_LAT'],
                                lon_0=self.site.location['GEO_LON'],
                                height_0=self.site.location['GEO_ALT'])
        cs_geo = cs_LENU.to_GEO()
        configured_variables = var_config.configured_variables

        var = self.add_variable(var_name='GEO_LAT', value=cs_geo['lat'],
                                configured_variables=configured_variables)
        var = self.add_variable(var_name='GEO_LON', value=cs_geo['lon'],
                                configured_variables=configured_variables)
        var = self.add_variable(var_name='GEO_ALT', value=cs_geo['height'],
                                configured_variables=configured_variables)

        if AACGM:
            cs_geo.ut = self['DATETIME'].value
            cs_aacgm = cs_geo.to_AACGM()
            var = self.add_variable(var_name='AACGM_LAT', value=cs_aacgm['lat'],
                                    configured_variables=configured_variables)
            var = self.add_variable(var_name='AACGM_LON', value=cs_aacgm['lon'],
                                    configured_variables=configured_variables)
        # var = self.add_variable(var_name='AACGM_ALT', value=cs_new['height'])

        if APEX:
            cs_geo.ut = self['DATETIME'].value
            cs_apex = cs_geo.to_APEX()
            var = self.add_variable(var_name='APEX_LAT', value=cs_apex['lat'],
                                    configured_variables=configured_variables)
            var = self.add_variable(var_name='APEX_LON', value=cs_apex['lon'],
                                    configured_variables=configured_variables)
        # var = self.add_variable(var_name='AACGM_ALT', value=cs_new['height'])
        pass

    def search_data_files(self, **kwargs):
        dt_fr = self.dt_fr
        dt_to = self.dt_to
        diff_days = dttool.get_diff_days(dt_fr, dt_to)
        day0 = dttool.get_start_of_the_day(dt_fr)
        for i in range(diff_days + 1):
            thisday = day0 + datetime.timedelta(days=i)
            if self.channel == 'color':
                tmp = 'color'
            else:
                tmp = self.channel + 'nm'

            initial_file_dir = self.data_root_dir / self.site / thisday.strftime('%Y') / thisday.strftime('%Y%m%d') / tmp

            file_patterns = []

            file_patterns.append(thisday.strftime('%y-%m-%d_'))

            # remove empty str
            file_patterns = [pattern for pattern in file_patterns if str(pattern)]

            search_pattern = '*' + '*'.join(file_patterns) + '*'

            done = super().search_data_files(
                initial_file_dir=initial_file_dir, search_pattern=search_pattern, recursive=True, allow_multiple_files=True)

            # Validate file paths

            if not done and self.allow_download:
                done = self.download_data()
                if done:
                    done = super().search_data_files(
                        initial_file_dir=initial_file_dir, search_pattern=search_pattern)
                else:
                    print('Cannot find files from the online database!')

        return done

    def download_data(self):
        raise NotImplementedError
        # if self.data_file_type == 'eiscat-hdf5':
        #     download_obj = self.downloader(dt_fr=self.dt_fr, dt_to=self.dt_to,
        #                                    sites=[self.site], kind_data='eiscat',
        #                                    data_file_root_dir=self.data_root_dir)
        # elif self.data_file_type == 'madrigal-hdf5':
        #     download_obj = self.downloader(dt_fr=self.dt_fr, dt_to=self.dt_to,
        #                                    sites=[self.site], kind_data='madrigal',
        #                                    data_file_root_dir=self.data_root_dir)
        # elif self.data_file_type == 'eiscat-mat':
        #     download_obj = self.downloader(dt_fr=self.dt_fr, dt_to=self.dt_to,
        #                                          sites=[self.site], kind_data='eiscat',
        #                                    data_file_root_dir=self.data_root_dir)
        # else:
        #     raise TypeError
        # return download_obj.done

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
    def site(self):
        return self._site

    @site.setter
    def site(self, value):
        if isinstance(value, str):
            if value == 'TRO':
                value = 'UHF'
            self._site = ASCSite(value)
        elif issubclass(value.__class__, SiteModel):
            self._site = value
        else:
            raise TypeError


class ASCSite(SiteModel):
    def __new__(cls, str_in, **kwargs):
        obj = super().__new__(cls, str_in, **kwargs)
        return obj

    def __init__(self, str_in, **kwargs):
        pass
        # site_info = {
        #     'TRO': {
        #         'name': 'Tromsø',
        #         'location': {
        #             'GEO_LAT': 69.58,
        #             'GEO_LON': 19.23,
        #             'GEO_ALT': 86 * 1e-3,   # in km
        #             'CGM_LAT': 66.73,
        #             'CGM_LOM': 102.18,
        #             'L (ground)': 6.45,
        #             'L (300km)':  6.70,
        #         },
        #     },
        #     'ESR': {
        #         'name': 'Longyearbyen',
        #         'location': {
        #             'GEO_LAT': 78.15,
        #             'GEO_LON': 16.02,
        #             'GEO_ALT': 445 * 1e-3,
        #             'CGM_LAT': 75.43,
        #             'CGM_LON': 110.68,
        #         }
        #     },
        # }
        #
        # if str_in in ['UHF', 'TRO']:
        #     self.name = site_info['TRO']['name'] + '-UHF'
        #     self.location = site_info['TRO']['location']
        # elif str_in == 'VHF':
        #     self.name = site_info['TRO']['name'] + '-VHF'
        #     self.location = site_info['TRO']['location']
        # elif str_in in ['ESR', 'LYB']:
        #     self.name = site_info['ESR']['name']
        #     self.location = site_info['ESR']['location']
        # else:
        #     raise NotImplementedError('The site ”{}" does not exist or has not been implemented!'.format(str_in))












