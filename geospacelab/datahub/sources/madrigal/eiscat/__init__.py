import datetime

import geospacelab.datahub as datahub
from geospacelab.datahub import DatabaseModel, FacilityModel, SiteModel
import geospacelab.config.preferences as prf
import geospacelab.toolbox.utilities.pydatetime as dttool
import geospacelab.datahub.sources.madrigal.eiscat.madrigal_eiscat_loader as default_loader
import geospacelab.datahub.sources.madrigal.eiscat.madrigal_eiscat_downloader as downloader
from geospacelab.datahub.sources.madrigal.eiscat.madrigal_eiscat_variable_config import items as default_var_configs
from geospacelab.datahub.sources.madrigal.eiscat.__utilities import *

default_dataset_attrs = {
    'database': 'Madrigal',
    'facility': 'EISCAT',
    'data_file_type': 'eiscat-hdf5',
    'data_file_ext': 'hdf5',
    'data_root_dir': prf.datahub_data_root_dir / 'Madrigal' / 'EISCAT' / 'analyzed',
    'download': True,
}

default_label_fields = ['database', 'facility', 'site', 'antenna', 'experiment']

default_variable_names = [
    'DATETIME', 'DATETIME_1', 'DATETIME_2',
    'magic_constant', 'r_SCangle', 'r_m0_1', 'r_m0_2',
    'az', 'el', 'P_Tx', 'height', 'range',
    'n_e', 'T_i', 'T_e', 'nu_i', 'v_i_los', 'comp_mix', 'comp_O_p',
    'n_e_err', 'T_i_err', 'T_e_err', 'nu_i_err', 'v_i_los_err', 'comp_mix_err', 'comp_O_p_err',
    'status', 'residual'
]

default_data_search_recursive = True


class Dataset(datahub.DatasetModel):
    def __init__(self, **kwargs):
        self.database = 'Madrigal'
        self.facility = 'EISCAT'
        self.site = ''
        self.antenna = ''
        self.experiment = ''
        self.pulse_code = ''
        self.scan_mode = ''
        self.modulation = ''
        self.data_file_type = ''
        self.affiliation = ''
        self.data_search_recursive = default_data_search_recursive
        self.download = False
        self._thisday = None

        super().__init__()
        kwargs = self._set_default_attrs(kwargs, default_dataset_attrs)
        self.config(**kwargs)

        self._set_default_variables(default_variable_names)

        self._validate_attrs()

    def _validate_attrs(self):
        if list(self.data_file_type):
            self.data_file_ext = self.data_file_type.split('-')[1]

    def label(self, **kwargs):
        self.label_fields = default_label_fields
        super().label()

    def load_data(self):
        self.check_data_files()

        if self.load_func is None:
            self.load_func = default_loader.select_loader(self.data_file_type)
            load_obj = self.load_func(self.data_file_paths)

            for var_name in self._variables.keys():
                self._variables[var_name] = load_obj.variables[var_name]
            self.site = load_obj.metadata['site_name']
            self.antenna = load_obj.metadata['antenna']
            self.pulse_code = load_obj.metadata['pulse_code']
            self.scan_mode = load_obj.metadata['scan_mode']
            self.experiment = load_obj.metadata['rawdata_path']
            self.affiliation = load_obj.metadata['affiliation']

    def search_data_files(self):
        dt_fr = self.dt_fr
        dt_to = self.dt_to
        diff_days = dttool.get_diff_days(dt_fr, dt_to)
        day0 = dttool.get_start_of_the_day(dt_fr)
        for i in range(diff_days + 1):
            thisday = day0 + datetime.timedelta(days=i)
            self._thisday = thisday
            initial_file_dir = self.data_root_dir / self.site / thisday.strftime('%Y')
            if not list(self.data_file_patterns):
                if self.data_file_type == 'eiscat-hdf5':
                    self.data_file_patterns.append('EISCAT')
                elif self.data_file_type == 'madrigal-hdf5':
                    self.data_file_patterns.append('MAD6400')
                elif self.data_file_type == 'eiscat-mat':
                    pass
                self.data_file_patterns.append(thisday.strftime('%Y-%m-%d'))
                self.data_file_patterns.append(self.modulation)
                self.data_file_patterns.append(self.antenna.lower())

            search_pattern = '*' + '*'.join(self.data_file_patterns) + '*'
            recursive = self.data_search_recursive
            done = super().search_data_files(
                initial_file_dir=initial_file_dir, search_pattern=search_pattern, recursive=recursive
            )

            if not done and self.download:
                done = self.download_data()
                if done:
                    done = super().search_data_files(
                initial_file_dir=initial_file_dir, search_pattern=search_pattern, recursive=recursive
            )
                else:
                    print('Cannot find files from the online database!')

    def download_data(self):
        if self.data_file_type == 'eiscat-hdf5':
            download_obj = downloader.Downloader(dt_fr=self.dt_fr, dt_to=self.dt_to,
                                                 sites=[self.site], kind_data='eiscat')
        elif self.data_file_type == 'madrigal-hdf5':
            download_obj = downloader.Downloader(dt_fr=self.dt_fr, dt_to=self.dt_to,
                                                 sites=[self.site], kind_data='madrigal')
        elif self.data_file_type == 'eiscat-mat':
            download_obj = downloader.Downloader(dt_fr=self.dt_fr, dt_to=self.dt_to,
                                                 sites=[self.site], kind_data='eiscat')
        return download_obj.done

    def set_variable(self, var_name, **kwargs):
        var_config_items = kwargs.pop('var_config_items', default_var_configs)
        var = super().set_variable(var_name=var_name, var_config_items=var_config_items)
        return var

    @property
    def database(self):
        return self._database

    @database.setter
    def database(self, value):
        self._database = DatabaseModel(value)

    @property
    def facility(self):
        return self._facility

    @facility.setter
    def facility(self, value):
        self._facility = FacilityModel(value)

    @property
    def site(self):
        return self._site

    @site.setter
    def site(self, value):
        self._site = Site(value)


class Site(SiteModel):
    def __new__(cls, str_in, **kwargs):
        obj = super().__new__(cls, str_in, **kwargs)
        return obj









