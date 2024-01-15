# Licensed under the BSD 3-Clause License
# Copyright (C) 2021 GeospaceLab (geospacelab)
# Author: Lei Cai, Space Physics and Astronomy, University of Oulu

__author__ = "Lei Cai"
__copyright__ = "Copyright 2021, GeospaceLab"
__license__ = "BSD-3-Clause License"
__email__ = "lei.cai@oulu.fi"
__docformat__ = "reStructureText"

import datetime
import numpy as np

from geospacelab.datahub.sources.madrigal.isr.millstonehill.basic.loader import Loader as default_Loader
from geospacelab.datahub.sources.madrigal.isr.millstonehill.downloader import Downloader as default_Downloader
import geospacelab.datahub.sources.madrigal.isr.millstonehill.basic.variable_config as var_config
from geospacelab.datahub.sources.madrigal import madrigal_database
from geospacelab.config import prf
from geospacelab import datahub
from geospacelab.datahub import SiteModel, DatabaseModel, FacilityModel
import geospacelab.toolbox.utilities.pybasic as basic
import geospacelab.toolbox.utilities.pydatetime as dttool
import geospacelab.toolbox.utilities.pylogging as mylog

default_dataset_attrs = {
    'kind': 'sourced',
    'database': madrigal_database,
    'facility': 'MillstoneHillISR',
    'exp_name_pattern': '',
    'exp_check': False,
    'data_file_type': '',
    'data_file_ext': 'hdf5',
    'data_root_dir': prf.datahub_data_root_dir / 'Madrigal' / 'MillstoneHill_ISR',
    'antenna': '',
    'pulse_code': '',
    'pulse_length': 0,
    'allow_download': True,
    'status_control': False,
    'residual_control': False,
    'beam_location': True,
    'data_search_recursive': True,
    'label_fields': ['database', 'facility', 'site', 'antenna', 'pulse_code', 'pulse_length'],
}

default_variable_names = [
    'DATETIME', 'AZ', 'AZ1', 'AZ2', 'EL', 'EL1', 'EL2', 'PULSE_LENGTH',
    'T_SYS', 'POWER_NORM', 'P_Tx', 'MODE_TYPE', 'POWER_LENGTH_F',
    'LAG_SPACING', 'IPP', 'f_Tx', 'v_PHASE_Tx', 'v_PHASE_Tx_err',
    'SCAN_TYPE', 'CYCN', 'POSN', 'RANGE_RES', 'RANGE',
    'SNR', 'RESIDUAL', 'STATUS', 'FIT_TYPE', 'FPI_QUALITY',
    'ACF_NORM', 'ACF_NORM_ERR', 'n_pp', 'n_pp_err', 'n_e',
    'n_e_err', 'T_i', 'T_i_err', 'T_r', 'T_r_err', 'T_e', 'T_e_err',
    'nu_i', 'nu_i_err', 'v_i_los', 'v_i_los_err', 'comp_H_p',
    'comp_H_p_err', 'comp_mix', 'comp_mix_err', 'v_DOP_los', 'v_DOP_los_err',
    'HEIGHT'
]

# default_data_search_recursive = True

default_attrs_required = ['antenna', 'pulse_code', 'pulse_length']


class Dataset(datahub.DatasetSourced):
    def __init__(self, **kwargs):
        kwargs = basic.dict_set_default(kwargs, **default_dataset_attrs)

        super().__init__(**kwargs)

        self.database = kwargs.pop('database', '')
        self.facility = kwargs.pop('facility', '')
        self.site = kwargs.pop('site', MillstoneHillSite('MillstoneHill'))
        self.antenna = kwargs.pop('antenna', '')
        self.experiment = kwargs.pop('experiment', '')
        self.exp_name_pattern = kwargs.pop('exp_name_pattern', '')
        self.exp_check = kwargs.pop('exp_check', False)
        self.pulse_code = kwargs.pop('pulse_code', '')
        self.pulse_length = kwargs.pop('pulse_length', '')
        self.data_file_type = kwargs.pop('data_file_type', '')
        self.affiliation = kwargs.pop('affiliation', '')
        self.allow_download = kwargs.pop('allow_download', True)
        self.beam_location = kwargs.pop('beam_location', True)
        self.metadata = None

        allow_load = kwargs.pop('allow_load', True)
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

        if self.exp_check:
            self.download_data()
            self.exp_check = False

    def label(self, **kwargs):
        label = super().label()
        return label

    def load_data(self, **kwargs):
        self.check_data_files(**kwargs)

        for file_path in self.data_file_paths:
            load_obj = self.loader(file_path, antenna=self.antenna, pulse_code=self.pulse_code, pulse_length=self.pulse_length)

            for var_name in self._variables.keys():
                self._variables[var_name].join(load_obj.variables[var_name])

                if var_name in ['n_pp', 'n_pp_err', 'n_e',
                                'n_e_err', 'T_i', 'T_i_err', 'T_r', 'T_r_err', 'T_e', 'T_e_err',
                                'nu_i', 'nu_i_err', 'v_i_los', 'v_i_los_err', 'comp_H_p',
                                'comp_H_p_err', 'comp_mix', 'comp_mix_err', 'v_DOP_los', 'v_DOP_los_err', ]:
                    if self.pulse_code == 'single pulse':
                        self._variables[var_name].visual.axis[1].lim = [90, 600]

            self.antenna = load_obj.metadata['antenna']
            self.pulse_code: str = load_obj.metadata['pulse_code']
            self.pulse_length = load_obj.metadata['pulse_length']
            self.metadata = load_obj.metadata
        if self.beam_location:
            self.calc_lat_lon()
            # self.select_beams(field_aligned=True)
        if self.time_clip:
            self.time_filter_by_range()
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
            'n_pp', 'n_pp_err', 'n_e',
            'n_e_err', 'T_i', 'T_i_err', 'T_r', 'T_r_err',
            'nu_i', 'nu_i_err', 'v_i_los', 'v_i_los_err', 'comp_H_p',
            'comp_H_p_err', 'comp_mix', 'comp_mix_err', 'v_DOP_los', 'v_DOP_los_err',
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
                mylog.StreamLogger.error("No field-aligned beams found!")
                raise ValueError

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

    def calc_lat_lon(self, AACGM=True):
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
        cs_new = cs_LENU.to_GEO()
        configured_variables = var_config.configured_variables

        var = self.add_variable(var_name='GEO_LAT', value=cs_new['lat'],
                                configured_variables=configured_variables)
        var = self.add_variable(var_name='GEO_LON', value=cs_new['lon'],
                                configured_variables=configured_variables)
        var = self.add_variable(var_name='GEO_ALT', value=cs_new['height'],
                                configured_variables=configured_variables)

        if AACGM:
            cs_new.ut = self['DATETIME'].value
            cs_new = cs_new.to_AACGM()
        var = self.add_variable(var_name='AACGM_LAT', value=cs_new['lat'],
                                configured_variables=configured_variables)
        var = self.add_variable(var_name='AACGM_LON', value=cs_new['lon'],
                                configured_variables=configured_variables)
        # var = self.add_variable(var_name='AACGM_ALT', value=cs_new['height'])
        pass

    def search_data_files(self, recursive=True, **kwargs):
        dt_fr = self.dt_fr
        dt_to = self.dt_to
        diff_days = dttool.get_diff_days(dt_fr, dt_to)
        day0 = dttool.get_start_of_the_day(dt_fr)
        for i in range(diff_days + 1):
            thisday = day0 + datetime.timedelta(days=i)
            initial_file_dir = self.data_root_dir / thisday.strftime('%Y') / thisday.strftime('%Y%m%d')

            if self.data_file_type == 'combined':
                file_patterns = ['MillstoneHill', self.data_file_type, thisday.strftime('%Y%m%d')]
            else:
                file_patterns = ['MillstoneHill', self.antenna, self.pulse_code.replace(' ', '_'), thisday.strftime('%Y%m%d')]

            # remove empty str
            file_patterns = [pattern for pattern in file_patterns if str(pattern)]

            search_pattern = '*' + '*'.join(file_patterns) + '*'
            if str(self.exp_name_pattern):
                search_pattern = '*' + self.exp_name_pattern.replace(' ', '-') + '*/' + search_pattern
                recursive = False

            done = super().search_data_files(
                initial_file_dir=initial_file_dir, search_pattern=search_pattern, recursive=recursive)


            # Validate file paths

            if not done and self.allow_download:
                done = self.download_data()
                if done:
                    done = super().search_data_files(
                        initial_file_dir=initial_file_dir, search_pattern=search_pattern, recursive=recursive)
                else:
                    print('The requested experiment does not exist in the online database!')
                    raise ValueError

            if len(done) > 1:
                if str(self.exp_name_pattern):
                    mylog.StreamLogger.error(
                        "Multiple data files detected! Check the files:")
                else:
                    mylog.StreamLogger.error(
                        "Multiple data files detected!" +
                        "Specify the experiment name by the keyword 'exp_name_pattern' if possible.")
                for fp in done:
                    mylog.simpleinfo.info(fp)
                raise KeyError

        return done

    def download_data(self):
        download_obj = self.downloader(
            dt_fr=self.dt_fr, dt_to=self.dt_to,
            data_file_root_dir=self.data_root_dir,
            file_type=self.data_file_type,
            exp_name_pattern=self.exp_name_pattern, dry_run=self.exp_check)
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
            self._site = MillstoneHillSite(value)
        elif issubclass(value.__class__, SiteModel):
            self._site = value
        else:
            raise TypeError


class MillstoneHillSite(SiteModel):
    def __new__(cls, str_in, **kwargs):
        obj = super().__new__(cls, str_in, **kwargs)
        return obj

    def __init__(self, str_in, **kwargs):
        self.name = 'MillstoneHill'
        self.location = {
            'GEO_LAT': 42.619,
            'GEO_LON': 288.51,
            'GEO_ALT': 0.146
        }












