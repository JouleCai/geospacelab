import datetime

import geospacelab.datahub as datahub
from geospacelab.datahub import DatabaseModel, FacilityModel, ProductModel
from geospacelab.datahub.sources.madrigal import madrigal_database
from geospacelab.config import prf
import geospacelab.toolbox.utilities.pydatetime as dttool
import geospacelab.toolbox.utilities.pybasic as basic
import geospacelab.toolbox.utilities.pylogging as mylog
from geospacelab.datahub.sources.madrigal.gnss.tecmap.loader import Loader as default_Loader
from geospacelab.datahub.sources.madrigal.gnss.tecmap.downloader import Downloader as default_Downloader
import geospacelab.datahub.sources.madrigal.gnss.tecmap.variable_config as var_config


default_dataset_attrs = {
    'database': madrigal_database,
    'facility': 'GNSS',
    'product': 'TEC',
    'data_file_type': 'TEC-MAP',
    'data_root_dir': prf.datahub_data_root_dir / 'Madrigal' / 'GNSS' / 'TEC',
    'allow_load': True,
    'allow_download': True,
    'data_search_recursive': False,
    'label_fields': ['database', 'facility', 'product', 'data_file_type'],
    'time_clip': True,
}

default_variable_names = ['DATETIME', 'GEO_LON', 'GEO_LAT', 'TEC_MAP']

# default_data_search_recursive = True

default_attrs_required = []


class Dataset(datahub.DatasetSourced):
    def __init__(self, **kwargs):
        kwargs = basic.dict_set_default(kwargs, **default_dataset_attrs)

        super().__init__(**kwargs)

        self.database = kwargs.pop('database', '')
        self.facility = kwargs.pop('facility', '')
        self.product = kwargs.pop('product', '')
        self.data_file_version = kwargs.pop('data_file_version', '')
        self.data_file_type = kwargs.pop('data_file_type', '')
        self.allow_download = kwargs.pop('allow_download', True)

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

    def label(self, **kwargs):
        label = super().label()
        return label

    def load_data(self, **kwargs):
        self.check_data_files(**kwargs)

        for file_path in self.data_file_paths:
            load_obj = self.loader(file_path=file_path, file_type=self.data_file_type)

            for var_name in self._variables.keys():
                if var_name in ['GEO_LAT', 'GEO_LON']:
                    self._variables[var_name].value = load_obj.variables[var_name]
                    continue
                self._variables[var_name].join(load_obj.variables[var_name])

            # self.select_beams(field_aligned=True)
        if self.time_clip:
            self.time_filter_by_range()

    def merge_tec_map(self, dt_fr, time_res=20*60):
        return

    def search_data_files(self, **kwargs):
        dt_fr = self.dt_fr
        dt_to = self.dt_to
        diff_days = dttool.get_diff_days(dt_fr, dt_to)
        dt0 = dttool.get_start_of_the_day(dt_fr)
        for i in range(diff_days + 1):
            thisday = dt0 + datetime.timedelta(days=i)

            initial_file_dir = kwargs.pop('initial_file_dir', None)
            if initial_file_dir is None:
                initial_file_dir = self.data_root_dir / thisday.strftime("%Y") / thisday.strftime('%Y%m%d')
            file_patterns = [self.data_file_type.replace('-', '_'), thisday.strftime("%Y%m%d"), self.data_file_version]
            # remove empty str
            file_patterns = [pattern for pattern in file_patterns if str(pattern)]

            search_pattern = '*' + '*'.join(file_patterns) + '*'

            done = super().search_data_files(
                initial_file_dir=initial_file_dir, search_pattern=search_pattern
            )

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
        download_obj = self.downloader(self.dt_fr, self.dt_to, file_type=self.data_file_type,
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











