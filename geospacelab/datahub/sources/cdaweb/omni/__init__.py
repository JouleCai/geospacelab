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
import functools

import geospacelab.datahub as datahub
from geospacelab.datahub import DatabaseModel, FacilityModel
from geospacelab.datahub.sources.cdaweb import cdaweb_database
from geospacelab.config import prf
import geospacelab.toolbox.utilities.pydatetime as dttool
import geospacelab.toolbox.utilities.pybasic as basic
import geospacelab.toolbox.utilities.pylogging as mylog
from geospacelab.datahub.sources.cdaweb.omni.loader import Loader as default_Loader
from geospacelab.datahub.sources.cdaweb.omni.downloader import Downloader as default_Downloader
import geospacelab.datahub.sources.cdaweb.omni.variable_config as var_config


default_dataset_attrs = {
    'facility': 'OMNI',
    'omni_type': 'omni2',
    'omni_res': '1min',
    'data_file_type': 'hres-cdf',
    'data_file_ext': 'cdf',
    'data_root_dir': prf.datahub_data_root_dir / 'CDAWeb' / 'OMNI',
    'allow_load': False,
    'allow_download': True,
    'data_search_recursive': False,
    'label_fields': ['database', 'facility', 'omni_type', 'omni_res'],
    'time_clip': True,
}

default_variable_names = ['DATETIME', 'SC_ID_IMF', 'SC_ID_PLS', 'IMF_PTS', 'PLS_PTS', 'PCT_INTERP',
                          'Timeshift', 'Timeshift_RMS',
                          'B_x_GSE', 'B_y_GSE', 'B_z_GSE',
                          'B_x_GSM', 'B_y_GSM', 'B_z_GSM', 'B_T_GSM', 'B_TOTAL',
                          'v_sw', 'v_x', 'v_y', 'v_z',
                          'n_p', 'T', 'p_dyn', 'E', 'beta', 'Ma_A', 'Ma_MSP',
                          'BSN_x', 'BSN_y', 'BSN_z',
                          'AE', 'AL', 'AU', 'SYM_H', 'SYM_D', 'ASY_H', 'ASY_D']

# default_data_search_recursive = True

default_attrs_required = ['omni_type', 'omni_res']


def _validate_IMF_cs(func):
    @functools.wraps(func)
    def wrapper(ds_omni, **kwargs):
        kwargs.setdefault('cs', 'GSM')
        kwargs.setdefault('to_radian', False)
        cs = kwargs['cs']
        if cs == 'GSM':
            kwargs.setdefault('Bx', ds_omni['B_x_GSM'].flatten())
            kwargs.setdefault('By', ds_omni['B_y_GSM'].flatten())
            kwargs.setdefault('Bz', ds_omni['B_z_GSM'].flatten())
        elif cs == 'GSE':
            kwargs.setdefault('Bx', ds_omni['B_x_GSE'].flatten())
            kwargs.setdefault('By', ds_omni['B_y_GSE'].flatten())
            kwargs.setdefault('Bz', ds_omni['B_z_GSE'].flatten())
        else:
            raise NotImplementedError
        result = func(ds_omni, **kwargs)

        if kwargs['to_radian']:
            unit = 'degree'
            unit_label = r'$^\circ$'
        else:
            unit = 'radiance'
            unit_label = r''
        result.unit = unit
        result.unit_label = unit_label
        result.depends = ds_omni['B_x_GSE'].depends
        if ds_omni.visual == 'on':
            result.visual.plot_config.style = '1P'
            result.visual.axis[1].unit = '@v.unit_label'
        return result

    return wrapper


class Dataset(datahub.DatasetSourced):
    def __init__(self, **kwargs):
        kwargs = basic.dict_set_default(kwargs, **default_dataset_attrs)

        super().__init__(**kwargs)

        self.database = cdaweb_database
        self.facility = kwargs.pop('facility', '')
        self.omni_type = kwargs.pop('omni_type', 'omni2')
        self.omni_res = kwargs.pop('omni_res', '1min')
        self.data_file_type = kwargs.pop('data_file_type','')
        self.allow_download = kwargs.pop('allow_download', True)
        self.force_download = kwargs.pop('force_download', False)
        self.download_dry_run = kwargs.pop('download_dry_run', False)

        self.metadata = None

        allow_load = kwargs.pop('allow_load', False)

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

    def label(self, **kwargs):
        label = super().label()
        return label

    def load_data(self, **kwargs):
        self.check_data_files(**kwargs)

        for file_path in self.data_file_paths:
            load_obj = self.loader(file_path, file_type=self.data_file_type)

            for var_name in self._variables.keys():
                self._variables[var_name].join(load_obj.variables[var_name])

            self.metadata = load_obj.metadata
            # self.select_beams(field_aligned=True)
        if self.time_clip:
            self.time_filter_by_range()

    def search_data_files(self, **kwargs):
        dt_fr = self.dt_fr
        dt_to = self.dt_to
        diff_months = dttool.get_diff_months(dt_fr, dt_to)
        dt0 = dttool.get_first_day_of_month(dt_fr)
        for i in range(diff_months + 1):
            thismonth = dttool.get_next_n_months(dt0, i)
            if self.omni_res in ['1min', '5min']:
                initial_file_dir = kwargs.pop('initial_file_dir', None)
                if initial_file_dir is None:
                    initial_file_dir = self.data_root_dir / \
                                       (self.omni_type.upper() + '_high_res_' + self.omni_res) / \
                                       '{:4d}'.format(thismonth.year)
                file_patterns = [
                    self.omni_res,
                    thismonth.strftime('%Y%m')
                ]
            elif self.omni_res == '1h':
                initial_file_dir = kwargs.pop('initial_file_dir', None)
                if initial_file_dir is None:
                    initial_file_dir = self.data_root_dir / \
                        'OMNI2_low_res_1h' / \
                        '{:4d}'.format(thismonth.year)
                file_patterns = [
                    'omni2',    
                    'mrg1hr',
                    thismonth.strftime('%Y')
                ]
            else:
                raise NotImplementedError
            # remove empty str
            search_pattern = kwargs.pop('search_pattern', None)
            if search_pattern is None:
                file_patterns = [pattern for pattern in file_patterns if str(pattern)]

                search_pattern = '*' + '*'.join(file_patterns) + '*'

            done = super().search_data_files(
                initial_file_dir=initial_file_dir, search_pattern=search_pattern)

            # Validate file paths

            if (not done and self.allow_download) or (self.force_download):
                done = self.download_data()
                if done:
                    done = super().search_data_files(
                        initial_file_dir=initial_file_dir, search_pattern=search_pattern)
                else:
                    print('Cannot find files from the online database!')
        if list(self.data_file_paths):
            self.data_file_paths = list(set(self.data_file_paths))
            self.data_file_paths.sort()
        return done

    def download_data(self):

        download_obj = self.downloader(
            dt_fr=self.dt_fr, dt_to=self.dt_to,
            time_res=self.omni_res, product=self.omni_type,
            root_dir_local=self.data_root_dir, 
            force_download=self.force_download,
            dry_run=self.download_dry_run,)
        return any(download_obj.done)

    # _validate_IMF_cs = staticmethod(_validate_IMF_cs)

    @_validate_IMF_cs
    def add_IMF_CA(self, cs='GSM', to_radian=False,  **kwargs):
        """
        IMF clock angle in the IMF By-Bz (x-y) plane in the GSE/GSM coordinate system.
        :return:
        """
        Bz = kwargs['Bz']
        By = kwargs['By']
        Bt = np.sqrt(By**2+Bz**2)
        sign_By = np.sign(By)
        sign_By[sign_By == 0] = 1.
        ca = sign_By * (np.pi/2 - np.arcsin(Bz / Bt))
        ca = np.mod(ca, 2 * np.pi)

        if not to_radian:
            ca = ca / np.pi * 180

        var_name = 'CA_' + cs
        var = self.add_variable(var_name, configured_variables=var_config.configured_variables, configured_variable_name='CA')
        var.value = ca.reshape(ca.size, 1)
        var.label = r'$\theta$'
        var.unit = 'degree'
        var.unit_label = r'($^\circ$)'
        self[var_name] = var
        return self[var_name]

    @_validate_IMF_cs
    def add_IMF_AZ(self, cs='GSM', to_radian=False,  **kwargs):
        """
        IMF azimuthal angle in the IMF Bx-By (x-y) plane in the GSE/GSM coordinate system.
        :return:
        """
        By = kwargs['By']
        Bx = kwargs['Bx']
        Bt = np.sqrt(By**2+Bx**2)
        sign_Bx = np.sign(Bx)
        sign_Bx[sign_Bx == 0] = 1.
        az = sign_Bx * (np.pi/2 - np.arcsin(By / Bt))
        az = np.mod(az, 2 * np.pi)

        if not to_radian:
            az = az / np.pi * 180

        var_name = 'AZ_' + cs
        var = self.add_variable(var_name, configured_variables=var_config.configured_variables, configured_variable_name='CA')
        var.value = az.reshape(az.size, 1)
        var.label = r'$\phi$'
        var.unit = 'degree'
        var.unit_label = r'($^\circ$)'
        self[var_name] = var
        return self[var_name]

    @_validate_IMF_cs
    def add_IMF_EL(self, cs='GSM', to_radian=False,  **kwargs):
        """
        IMF elevation angle in the IMF Bz-Bx (x-y) plane in the GSE/GSM coordinate system.
        :return:
        """
        Bx = kwargs['Bx']
        Bz = kwargs['Bz']
        Bt = np.sqrt(Bx ** 2 + Bz ** 2)
        sign_Bz = np.sign(Bz)
        sign_Bz[sign_Bz == 0] = 1.
        el = sign_Bz * (np.pi / 2 - np.arcsin(Bx / Bt))
        el = np.mod(el, 2 * np.pi)

        if not to_radian:
            el = el / np.pi * 180

        var_name = 'EL_' + cs
        var = self.add_variable(var_name, configured_variables=var_config.configured_variables, configured_variable_name='CA')
        var.value = el.reshape(el.size, 1)
        var.label = r'$\alpha$'
        var.unit = 'degree'
        var.unit_label = r'($^\circ$)'
        self[var_name] = var
        return self[var_name]

    def add_NCF(self):
        """
        Solar wind-magnetosphere coupling function by Newell et al., JGR, 2007.
        :return:
        """
        if 'CA_GSM' not in self.keys():
            self.add_IMF_CA()

        ca = self['CA_GSM'].value
        ca = ca * np.pi / 180
        v = self['v_sw'].value
        BT = self['B_T_GSM'].value
        ncf = np.abs(v)**(4/3) * np.abs(BT)**(2/3) * np.abs(np.sin(ca*0.5))**(8/3)

        self['NCF'] = var_config.configured_variables['NCF'].clone()
        self['NCF'].value = ncf

        return self['NCF']

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













