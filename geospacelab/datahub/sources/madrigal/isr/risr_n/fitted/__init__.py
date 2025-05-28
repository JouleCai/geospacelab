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
import h5py

from geospacelab.datahub.sources.madrigal.isr.risr_n.fitted.loader import Loader as default_Loader
from geospacelab.datahub.sources.madrigal.isr.risr_n.fitted.downloader import Downloader as default_Downloader
import geospacelab.datahub.sources.madrigal.isr.risr_n.fitted.variable_config as var_config
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
    'facility': 'RISR-N',
    'exp_name_pattern': [],
    'exp_check': False,
    'data_file_ext': ['h5', 'hdf5'],
    'data_root_dir': prf.datahub_data_root_dir / 'Madrigal' / 'RISR-N',
    'allow_download': True,
    'status_control': False,
    'residual_control': False,
    'add_AACGM': True,
    'add_APEX': False,
    'data_search_recursive': True,
    'label_fields': ['database', 'facility', 'site', 'data_file_type'],
}

default_variable_names = [
    'DATETIME', 'AZ', 'EL', 'PULSE_LENGTH',
    'P_Tx', 'n_e', 'n_e_err', 'T_i', 'T_i_err', 'T_e', 'T_e_err',
    'v_i_los', 'v_i_los_err', 'comp_mix', 'comp_mix_err',
    'HEIGHT', 'RANGE', 'CGM_LAT', 'CGM_LON', 'BEAM_ID', 'CHISQ'
]

# default_data_search_recursive = True

default_attrs_required = []

pulse_code_dict = {
    'alternating code': 'AC',
    'long pulse': 'LP',
}


class Dataset(datahub.DatasetSourced):
    def __init__(self, **kwargs):
        kwargs = basic.dict_set_default(kwargs, **default_dataset_attrs)

        super().__init__(**kwargs)

        self.database = kwargs.pop('database', '')
        self.facility = kwargs.pop('facility', '')
        self.site = kwargs.pop('site', RISR('RISR_N'))
        self.experiment = kwargs.pop('experiment', '')
        self.experiment_ids = kwargs.pop('exp_ids', [])
        self.exp_name_pattern = kwargs.pop('exp_name_pattern', [])
        self.integration_time = kwargs.pop('integration_time', None)
        self.exp_check = kwargs.pop('exp_check', False)
        self.affiliation = kwargs.pop('affiliation', '')
        self.pulse_code = kwargs.pop('pulse_code', 'alternating code')
        self.allow_download = kwargs.pop('allow_download', True)
        self.beam_id = kwargs.pop('beam_id', None)
        self.beam_az = kwargs.pop('beam_az', None)
        self.beam_el = kwargs.pop('beam_el', None)
        self.beams = kwargs.pop('beams', None)
        self.show_beams = kwargs.pop('show_beams', False)
        self.add_AACGM = kwargs.pop('add_AACGM', True)
        self.add_APEX = kwargs.pop('add_APEX', False)
        self.metadata = None

        self.data_file_type = pulse_code_dict[self.pulse_code.lower()] + '_fitted'

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
            self.download_data(dry_run=True)

    def label(self, **kwargs):
        label = super().label()
        return label

    def load_data(self, **kwargs):
        self.check_data_files(**kwargs)

        for file_path in self.data_file_paths:
            load_obj = self.loader(file_path, beam_az=self.beam_az, beam_el=self.beam_el, beam_id=self.beam_id)

            for var_name in self._variables.keys():
                self._variables[var_name].join(load_obj.variables[var_name])
            self.metadata = load_obj.metadata

        if self.show_beams:
            load_obj.list_beams()
        self.beams = load_obj.beams
        self.beam_id = load_obj.beam_id
        self.beam_az = load_obj.beam_az
        self.beam_el = load_obj.beam_el

        # if self.add_APEX or self.add_AACGM:
        self.calc_lat_lon()

        if self['HEIGHT'].value is None:
            self['HEIGHT'] = self['GEO_ALT']

        if self.time_clip:
            self.time_filter_by_range()

    def calc_lat_lon(self):
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

    def search_data_files(self, recursive=True, **kwargs):
        dt_fr = self.dt_fr
        dt_to = self.dt_to
        done = False
        if list(self.experiment_ids):
            initial_file_dir = self.data_root_dir
            
            for exp_id in self.experiment_ids:
                search_pattern = f"*EID-{exp_id}*/*{self.data_file_type}*"
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
                    if isinstance(self.exp_name_pattern, (str, list)):
                        mylog.StreamLogger.warning(
                            "Multiple data files detected! Check the files:")
                    else:
                        mylog.StreamLogger.warning(
                            "Multiple data files detected!" +
                            "Specify the experiment name by the keyword 'exp_name_pattern' if possible.")
                    for fp in done:
                        mylog.simpleinfo.info(fp)

        else:
            diff_years = dt_to.year - dt_fr.year
            for ny in range(diff_years + 1):
                initial_file_dir = self.data_root_dir / str(dt_to.year + ny)
                search_pattern = "*EID-*/"
                exp_dirs = list(initial_file_dir.glob(search_pattern))

                if not list(exp_dirs) and self.allow_download:
                    self.download_data()
                    
                def dir_parser(dirs):
                    dirs_out = []
                    for d in dirs:
                        rc = re.compile(r'_([\d]{14})')
                        res = rc.findall(str(d))
                        dt_0 = datetime.datetime.strptime(res[0], "%Y%m%d%H%M%S")
                        dt_1 = datetime.datetime.strptime(res[1], "%Y%m%d%H%M%S")

                        if dt_1 < dt_fr:
                            continue
                        if dt_0 > dt_to:
                            continue
                        dirs_out.append(d)
                    return dirs_out
                file_dirs = dir_parser(exp_dirs)

                if not list(file_dirs) and self.allow_download:
                    self.download_data() 
                
                for fd in file_dirs:
                    if isinstance(self.exp_name_pattern, list):
                        p = '.*'.join(self.exp_name_pattern)
                    elif isinstance(self.exp_name_pattern, str):
                        p = self.exp_name_pattern
                    else:
                        p = '.*'
                    rc = re.compile(p)
                    res = rc.search(str(fd))
                    if res is None:
                        continue

                    file_patterns = ['RISR-N', self.data_file_type]

                    # remove empty str
                    file_patterns = [pattern for pattern in file_patterns if str(pattern)]

                    search_pattern = '*' + '*'.join(file_patterns) + '*'

                    recursive = False

                    done = super().search_data_files(
                        initial_file_dir=fd, search_pattern=search_pattern,
                        allow_multiple_files=True, recursive=recursive)


                    # Validate file paths
                    if not done and self.allow_download:
                        done = self.download_data()
                        if done:
                            done = super().search_data_files(
                                initial_file_dir=initial_file_dir,
                                search_pattern=search_pattern,
                                allow_multiple_files=True, recursive=recursive
                            )
                        else:
                            print('The requested experiment does not exist in the online database!')

                    if len(done) > 1:
                        if isinstance(self.exp_name_pattern, (str, list)):
                            mylog.StreamLogger.warning(
                                "Multiple data files detected! Check the files:")
                        else:
                            mylog.StreamLogger.warning(
                                "Multiple data files detected!" +
                                "Specify the experiment name by the keyword 'exp_name_pattern' if possible.")
                        for fp in done:
                            mylog.simpleinfo.info(fp)

        self._select_file_by_time_resolution()
        import natsort
        self.data_file_paths = natsort.natsorted(self.data_file_paths, reverse=False)
        return done

    def _select_file_by_time_resolution(self, file_paths=None):
        def show_option():
            mylog.simpleinfo.info("Options for the integration time [s]: ")
            ts = np.unique(itg_times)
            for t in ts:
                mylog.simpleinfo.info(f"{t}")

        if file_paths is None:
            file_paths = self.data_file_paths
        itg_times = []
        for file_path in file_paths:
            with (h5py.File(file_path, 'r') as fh5):
                beam_str = list(fh5['Data']['Array Layout'].keys())[0]
                diff_t = np.nanmedian(np.diff(fh5['Data']['Array Layout'][beam_str]['timestamps'][::]))
                itg_time = np.floor(diff_t/60) * 60
                itg_times.append(np.median(itg_time))

        if self.integration_time is None:
            mylog.StreamLogger.error(f"The integration time was not set. Reset is required!")
            show_option()
            raise AttributeError
            
        if self.integration_time not in itg_times:
            mylog.StreamLogger.error(f"The integration time is not found. Reset is required!")
            show_option()
            raise AttributeError 
        
        ind = np.where(np.array(itg_times, dtype=np.float32) == self.integration_time)[0]

        file_paths = [file_paths[i] for i in ind]
        mylog.simpleinfo.info(f'The following files are selected with the integration time {self.integration_time} [s]:')
        for file_path in file_paths:
            mylog.simpleinfo.info(file_path)

        self.data_file_paths = [self.data_file_paths[i] for i in ind]

    def download_data(self, dry_run=False):
        include_exp_name_patterns = [self.exp_name_pattern] if isinstance(self.exp_name_pattern, list) else []
        download_obj = self.downloader(
            dt_fr=self.dt_fr, dt_to=self.dt_to,
            pulse_code=self.pulse_code,
            dry_run=dry_run,
            data_file_root_dir=self.data_root_dir,
            include_exp_ids=self.experiment_ids,
            include_exp_name_patterns=include_exp_name_patterns)
        return download_obj.data_file_paths


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
            self._site = RISR(value)
        elif issubclass(value.__class__, SiteModel):
            self._site = value
        else:
            raise TypeError


class RISR(SiteModel):
    def __new__(cls, str_in, **kwargs):
        obj = super().__new__(cls, str_in, **kwargs)
        return obj

    def __init__(self, str_in, **kwargs):
        self.name = 'Resolute Bay North IS Radar'
        self.location = {
            'GEO_LAT': 74.72955,
            'GEO_LON': 256.09424,
            'GEO_ALT': 0.145
        }












