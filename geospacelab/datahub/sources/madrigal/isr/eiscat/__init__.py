# Licensed under the BSD 3-Clause License
# Copyright (C) 2021 GeospaceLab (geospacelab)
# Author: Lei Cai, Space Physics and Astronomy, University of Oulu

__author__ = "Lei Cai"
__copyright__ = "Copyright 2021, GeospaceLab"
__license__ = "BSD-3-Clause License"
__email__ = "lei.cai@oulu.fi"
__docformat__ = "reStructureText"


import datetime
import re

import numpy as np

import geospacelab.datahub as datahub
from geospacelab.datahub import DatabaseModel, FacilityModel, SiteModel
from geospacelab.datahub.sources.madrigal import madrigal_database
from geospacelab.config import pref as prf
import geospacelab.toolbox.utilities.pydatetime as dttool
import geospacelab.toolbox.utilities.pybasic as basic
from geospacelab.datahub.sources.madrigal.isr.eiscat.loader import Loader as default_Loader
from geospacelab.datahub.sources.madrigal.isr.eiscat.downloader import Downloader as default_Downloader
import geospacelab.datahub.sources.madrigal.isr.eiscat.variable_config as var_config
from geospacelab.datahub.sources.madrigal.isr.eiscat.utilities import *

default_dataset_attrs = {
    'kind': 'sourced',
    'database': madrigal_database,
    'facility': 'EISCAT',
    'data_file_type': 'madrigal-hdf5',
    'data_file_ext': 'hdf5',
    'data_root_dir': prf.datahub_data_root_dir / 'Madrigal' / 'EISCAT' / 'Analyzed',
    'allow_download': True,
    'status_control': False,
    'rasidual_contorl': False,
    'data_search_recursive': True,
    'add_AACGM': True,
    'add_APEX': False,
    'label_fields': ['database', 'facility', 'site', 'antenna', 'experiment'],
}

default_variable_names = [
    'DATETIME', 'DATETIME_1', 'DATETIME_2',
    'AZ', 'EL', 'P_Tx', 'HEIGHT', 'RANGE', 'P_Tx', 'T_SYS_1', 'T_SYS_2',
    'n_e', 'T_i', 'T_e', 'nu_i', 'v_i_los', 'comp_mix', 'comp_O_p',
    'n_e_err', 'T_i_err', 'T_e_err', 'nu_i_err', 'v_i_los_err', 'comp_mix_err', 'comp_O_p_err',
    'STATUS', 'RESIDUAL'
]

# default_data_search_recursive = True

default_attrs_required = ['site', 'antenna',]


class Dataset(datahub.DatasetSourced):
    def __init__(self, **kwargs):
        kwargs = basic.dict_set_default(kwargs, **default_dataset_attrs)

        super().__init__(**kwargs)

        self.database = kwargs.pop('database', '')
        self.facility = kwargs.pop('facility', '')
        self.site = kwargs.pop('site', '')
        self.antenna = kwargs.pop('antenna', '')
        self.experiment = kwargs.pop('experiment', '')
        self.experiment_ids = kwargs.pop('exp_ids', [])
        self.pulse_code = kwargs.pop('pulse_code', '')
        self.scan_mode = kwargs.pop('scan_mode', '')
        self.modulation = kwargs.pop('modulation', '')
        self.data_file_type = kwargs.pop('data_file_type', '')
        self.affiliation = kwargs.pop('affiliation', '')
        self.allow_download = kwargs.pop('allow_download', True)
        self.gate_num = kwargs.pop('gate_num', None)
        self.metadata = {}
        self.add_AACGM = kwargs.pop('add_AACGM', True)
        self.add_APEX = kwargs.pop('add_APEX', False)

        allow_load = kwargs.pop('allow_load', False)
        self.status_control = kwargs.pop('status_control', False)
        self.residual_control = kwargs.pop('residual_control', False)

        # self.config(**kwargs)

        if self.loader is None:
            self.loader = default_Loader
        if self.downloader is None:
            self.downloader = default_Downloader

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
            if (self.load_mode == 'AUTO') and (self.data_file_type=='eiscat-mat'):
                raise AttributeError

    def label(self, **kwargs):
        label = super().label()
        return label

    def load_data(self, **kwargs):
        self.check_data_files(**kwargs)

        for file_path in self.data_file_paths:
            load_obj = self.loader(file_path, file_type=self.data_file_type, gate_num=self.gate_num)
            if self.gate_num is None:
                self.gate_num = load_obj.gate_num

            for var_name in self._variables.keys():
                self._variables[var_name].join(load_obj.variables[var_name])

            self.site = load_obj.metadata['site_name']
            self.antenna = load_obj.metadata['antenna']
            self.pulse_code = load_obj.metadata['pulse_code']
            self.scan_mode = load_obj.metadata['scan_mode']
            self.modulation = load_obj.metadata['modulation']
            rawdata_path = load_obj.metadata['rawdata_path']
            self.experiment = rawdata_path.split('/')[-1].split('@')[0]
            self.affiliation = load_obj.metadata['affiliation']
            self.metadata = load_obj.metadata

        if self.add_AACGM or self.add_APEX:
            self.calc_lat_lon()
            # self.select_beams(field_aligned=True)
        if self.time_clip:
            self.time_filter_by_range()

        inds_cmb = np.argsort(self['DATETIME'].flatten())
        if any(np.diff(np.array(inds_cmb)) < 0):
            for var_name in self.keys():
                self[var_name].value = self[var_name].value[inds_cmb, :]

        if self.status_control:
            self.status_mask()
        if self.residual_control:
            self.residual_mask()

    def outlier_mask(self, condition, fill_value=None):
        """
        Mask outliers of the 2D variables (Ne, Te, Ti, ...) depending on condition.
        :param condition: a bool array with the same dimension as the 2D variables
        :type condition: np.ndarray, bool
        :param fill_value: the value filled as the outlier, [np.nan]
        :type fill_value: np.nan or float
        """
        var_2d_names = [
            'n_e', 'T_i', 'T_e', 'nu_i', 'v_i_los',
            'comp_mix', 'comp_O_p', 'n_e_err', 'T_i_err', 'T_e_err',
            'nu_i_err', 'v_i_los_err', 'comp_mix_err', 'comp_O_p_err'
        ]
        for key in self.keys():
            if key in var_2d_names:
                self[key].value[condition] = np.nan

    def status_mask(self, bad_status=None):
        """
        Mask the 2D variables depending on status.
        :param bad_status: a list of the status flags, [2, 3].
        :type bad_status: list
        """
        if bad_status is None:
            bad_status = [2, 3]
        for flag in bad_status:
            condition = self['STATUS'].value == flag
            self.outlier_mask(condition)

    def residual_mask(self, residual_lim=None):
        """
        Mask the 2D variables depending on residual
        :param residual_lim: the lower limit of the bad residual, [10].
        :type residual_lim: float
        """
        if residual_lim is None:
            residual_lim = 10

        condition = self['RESIDUAL'].value > residual_lim
        self.outlier_mask(condition)

    def select_beams(self, field_aligned=False, az_el_pairs=None, error_az=2, error_el=2):
        if field_aligned:
            if az_el_pairs is not None:
                raise AttributeError("The parameters field_aligned and az_el_pairs cannot be set at the same time!")
            if self.site != 'UHF':
                raise AttributeError("Only UHF can be applied.")

        az = self['AZ'].value.flatten() % 360.
        el = self['EL'].value.flatten()
        if field_aligned:
            inds = np.where(((np.abs(az - 188.6) <= error_az) & (np.abs(el-77.7) <= error_el)))[0]
            if not list(inds):
                mylog.StreamLogger.info("No field-aligned beams found!")
                return
        elif isinstance(az_el_pairs, list):
            inds = []
            for az1, el1 in az_el_pairs:
                az1 = az1 % 360.
                ind_1 = np.where(((np.abs(az - az1) <= error_az) & (np.abs(el-el1) <= error_el)))[0]
                ind_2 = np.where(((np.abs(az - 360. - az1) <= error_az) & (np.abs(el-el1) <= error_el)))[0] 
                ind_1 = np.append(ind_1, ind_2)
                if not list(ind_1):
                    mylog.StreamLogger.error("Cannot find the beam with az={:f} and el={:f}".format(az1, el1))
                    raise ValueError
                inds.extend(ind_1)
            inds = np.sort(np.unique(np.array(inds))).tolist()
        else:
            raise ValueError
        self.time_filter_by_inds(inds)

    def calc_lat_lon(self, ):
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

        if self.add_AACGM:
            cs_geo.ut = self['DATETIME'].value
            cs_aacgm = cs_geo.to_AACGM()
            var = self.add_variable(var_name='AACGM_LAT', value=cs_aacgm['lat'],
                                    configured_variables=configured_variables)
            var = self.add_variable(var_name='AACGM_LON', value=cs_aacgm['lon'],
                                    configured_variables=configured_variables)
        # var = self.add_variable(var_name='AACGM_ALT', value=cs_new['height'])

        if self.add_APEX:
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
        done = False
        if not list(self.experiment_ids):
            diff_days = dttool.get_diff_days(dt_fr, dt_to)
            day0 = dttool.get_start_of_the_day(dt_fr)
            for i in range(diff_days + 1):
                thisday = day0 + datetime.timedelta(days=i)
                initial_file_dir = self.data_root_dir / self.site / thisday.strftime('%Y')

                file_patterns = []
                if self.data_file_type == 'eiscat-hdf5':
                    file_patterns.append('EISCAT')
                elif self.data_file_type == 'madrigal-hdf5':
                    file_patterns.append('MAD6400')
                else:
                    raise NotImplementedError
                file_patterns.append(thisday.strftime('%Y-%m-%d'))
                if str(self.pulse_code):
                    file_patterns.append(self.pulse_code)
                if str(self.modulation):
                    file_patterns.append(self.modulation)
                file_patterns.append(self.antenna.lower())

                # remove empty str
                file_patterns = [pattern for pattern in file_patterns if str(pattern)]

                search_pattern = '*' + '*'.join(file_patterns) + '*'
                done = super().search_data_files(
                    initial_file_dir=initial_file_dir,
                    search_pattern=search_pattern, allow_multiple_files=True)

                # Validate file paths
                if not done and self.allow_download:
                    done = self.download_data()
                    if done:
                        done = super().search_data_files(
                            initial_file_dir=initial_file_dir,
                            search_pattern=search_pattern,
                            allow_multiple_files=True
                        )
                    else:
                        print('Cannot find files from the online database!')
        else:
            initial_file_dir = self.data_root_dir
            for exp_id in self.experiment_ids:
                file_patterns = []
                if self.data_file_type == 'eiscat-hdf5':
                    file_patterns.append('EISCAT')
                elif self.data_file_type == 'madrigal-hdf5':
                    file_patterns.append('MAD')
                else:
                    raise NotImplementedError
                file_patterns.append(thisday.strftime('%Y-%m-%d'))
                if str(self.pulse_code):
                    file_patterns.append(self.pulse_code)
                if str(self.modulation):
                    file_patterns.append(self.modulation)
                file_patterns.append(self.antenna.lower())
                # remove empty str
                file_patterns = [pattern for pattern in file_patterns if str(pattern)]
                search_pattern = f"*EID-{exp_id}*/*{'*'.join(file_patterns)}*"
                done = super().search_data_files(
                    initial_file_dir=initial_file_dir,
                    search_pattern=search_pattern, recursive=True, allow_multiple_files=True
                )

                if not done and self.allow_download:
                    done = self.download_data()
                    if done:
                        done = super().search_data_files(
                            initial_file_dir=initial_file_dir,
                            search_pattern=search_pattern, recursive=True, allow_multiple_files=True
                        )
                    else:
                        print('The requested experiment (ID: {}) does not exist in the online database!'.format(exp_id))
                if len(done) > 1:
                    mylog.StreamLogger.warning(
                        "Multiple data files detected! " +
                        "Specify the experiment pulse code and modulation may constrain the searching condition.")
                    for fp in done:
                        mylog.simpleinfo.info(str(fp))
        self._check_multiple_files()

        return done

    def _check_multiple_files(self):
        file_paths = self.data_file_paths
        exp_ids = []
        for fp in file_paths:
            rc = re.compile(r"EID\-([\d]+)")
            res = rc.search(str(fp))
            exp_ids.append(res.groups()[0])
        exp_ids_unique = [eid for eid in np.unique(exp_ids)]

        file_paths_new = []
        for eid in exp_ids_unique:
            inds_id = np.where(np.array(exp_ids)==eid)[0]
            fps_sub = []
            for ii in inds_id:
                fp = file_paths[ii]
                rc = re.compile(r".*_([\d]{8}T[\d]{6}).*_([\d]{8}T[\d]{6}).*[\d]{4}\-[\d]{2}\-[\d]{2}_([\w.]+)@.*")
                res = rc.search(str(fp))
                dt_0 = datetime.datetime.strptime(res.groups()[0], '%Y%m%dT%H%M%S')
                dt_1 = datetime.datetime.strptime(res.groups()[1], '%Y%m%dT%H%M%S')
                if (dt_0 >= self.dt_to) or (dt_1<=self.dt_fr):
                    continue
                if str(self.pulse_code):
                    if self.pulse_code not in res.groups()[2].lower():
                        continue
                if str(self.modulation):
                    if self.modulation not in res.groups()[2].lower():
                        continue
                if '_v' in res.groups()[2].lower():
                    continue
                fps_sub.extend([fp])
            if len(fps_sub) > 1:
                mylog.StreamLogger.warning("Multiple data files for a single experiment detected!")
                # for fp in fps_sub:
                #     mylog.simpleinfo.info(str(fp))
                # fps_sub = fps_sub[0]
            file_paths_new.extend(fps_sub)
        self.data_file_paths = file_paths_new


    def download_data(self):
        if self.data_file_type == 'eiscat-hdf5':
            download_obj = self.downloader(dt_fr=self.dt_fr, dt_to=self.dt_to,
                                           antennas=[self.antenna], kind_data='eiscat',
                                           data_file_root_dir=self.data_root_dir,
                                           exclude_file_type_patterns=['pp']
            )
        elif self.data_file_type == 'madrigal-hdf5':
            download_obj = self.downloader(dt_fr=self.dt_fr, dt_to=self.dt_to,
                                           antennas=[self.antenna], kind_data='madrigal',
                                           data_file_root_dir=self.data_root_dir,
                                           exclude_file_type_patterns=['pp']
                                           )
        else:
            raise TypeError
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
            self._site = EISCATSite(value)
        elif issubclass(value.__class__, SiteModel):
            self._site = value
        else:
            raise TypeError


class EISCATSite(SiteModel):
    def __new__(cls, str_in, **kwargs):
        obj = super().__new__(cls, str_in, **kwargs)
        return obj

    def __init__(self, str_in, **kwargs):
        site_info = {
            'TRO': {
                'name': 'Tromsø',
                'location': {
                    'GEO_LAT': 69.58,
                    'GEO_LON': 19.23,
                    'GEO_ALT': 86 * 1e-3,   # in km
                    'CGM_LAT': 66.73,
                    'CGM_LOM': 102.18,
                    'L (ground)': 6.45,
                    'L (300km)':  6.70,
                },
            },
            'ESR': {
                'name': 'Longyearbyen',
                'location': {
                    'GEO_LAT': 78.15,
                    'GEO_LON': 16.02,
                    'GEO_ALT': 445 * 1e-3,
                    'CGM_LAT': 75.43,
                    'CGM_LON': 110.68,
                }
            },
        }

        if str_in in ['UHF', 'TRO']:
            self.name = site_info['TRO']['name'] + '-UHF'
            self.location = site_info['TRO']['location']
        elif str_in == 'VHF':
            self.name = site_info['TRO']['name'] + '-VHF'
            self.location = site_info['TRO']['location']
        elif str_in in ['ESR', 'LYB']:
            self.name = site_info['ESR']['name']
            self.location = site_info['ESR']['location']
        else:
            raise NotImplementedError('The site ”{}" does not exist or has not been implemented!'.format(str_in))












