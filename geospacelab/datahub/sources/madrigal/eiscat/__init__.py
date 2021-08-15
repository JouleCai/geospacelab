import datetime
import numpy as np

import geospacelab.datahub as datahub
from geospacelab.datahub import DatabaseModel, FacilityModel, SiteModel
from geospacelab.datahub.sources.madrigal import madrigal_database
from geospacelab import preferences as prf
import geospacelab.toolbox.utilities.pydatetime as dttool
import geospacelab.toolbox.utilities.pybasic as basic
import geospacelab.datahub.sources.madrigal.eiscat._loader as default_loader
import geospacelab.datahub.sources.madrigal.eiscat._downloader as downloader
import geospacelab.datahub.sources.madrigal.eiscat._variable_config as var_config
from geospacelab.datahub.sources.madrigal.eiscat.utilities import *

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
    'az', 'el', 'P_Tx', 'height', 'range',
    'n_e', 'T_i', 'T_e', 'nu_i', 'v_i_los', 'comp_mix', 'comp_O_p',
    'n_e_err', 'T_i_err', 'T_e_err', 'nu_i_err', 'v_i_los_err', 'comp_mix_err', 'comp_O_p_err',
    'status', 'residual'
]

default_data_search_recursive = True

default_attrs_required = ['site', 'antenna', 'modulation']


class Dataset(datahub.DatasetModel):
    def __init__(self, **kwargs):
        self.database = madrigal_database
        self.facility = 'EISCAT'
        self.site = None
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
        self.metadata = None

        load_data = kwargs.pop('load_data', False)

        super().__init__()
        kwargs = basic.dict_set_default(kwargs, **default_dataset_attrs)
        self.config(**kwargs)

        self._set_default_variables(
            default_variable_names,
            configured_variables=var_config.get_default_configured_variables()
        )

        self._validate_attrs()

        if load_data:
            self.load_data()

    def _validate_attrs(self):
        for attr_name in default_attrs_required:
            attr = getattr(self, attr_name)
            if not str(attr):
                mylog.StreamLogger.warning("The parameter {} is required before loading data!".format(attr_name))

        if list(self.data_file_type):
            self.data_file_ext = self.data_file_type.split('-')[1]

    def label(self, **kwargs):
        self.label_fields = default_label_fields
        label = super().label()
        return label

    def load_data(self, **kwargs):
        self.check_data_files(**kwargs)

        if self.load_func is None:
            self.load_func = default_loader.select_loader(self.data_file_type)

            for file_path in self.data_file_paths:
                load_obj = self.load_func(file_path)

                for var_name in self._variables.keys():
                    self._variables[var_name].join(load_obj.variables[var_name])

                self.site = load_obj.metadata['site_name']
                self.antenna = load_obj.metadata['antenna']
                self.pulse_code = load_obj.metadata['pulse_code']
                self.scan_mode = load_obj.metadata['scan_mode']
                rawdata_path = load_obj.metadata['rawdata_path']
                self.experiment = rawdata_path.split('/')[-1].split('@')[0]
                self.affiliation = load_obj.metadata['affiliation']
                self.metadata = load_obj.metadata
            self.calc_lat_lon()

    def filter_field_aligned(self):
        pass

    def calc_lat_lon(self):
        from geospacelab.cs._geo import GEO, LENU
        az = self['az'].value
        el = self['el'].value
        range = self['range'].value
        az = np.tile(az, range.shape)  # make az, el, range in the same shape
        el = np.tile(el, range.shape)
        cs_LENU = LENU(coords={'az': az, 'el': el, 'range': range},
                       lat_0=self.site.location['GEO_LAT'],
                       lon_0=self.site.location['GEO_LON'],
                       height_0=self.site.location['GEO_ALT'],
                       kind='sph')
        cs_new = cs_LENU.to_GEO()
        self.add_variable(var_name='GEO_LAT', value=cs_new['lat'])
        self.add_variable(var_name='GEO_LON', value=cs_new['lon'])
        self.add_variable(var_name='GEO_ALT', value=cs_new['height'])
        pass

    def search_data_files(self, **kwargs):
        dt_fr = self.dt_fr
        dt_to = self.dt_to
        diff_days = dttool.get_diff_days(dt_fr, dt_to)
        day0 = dttool.get_start_of_the_day(dt_fr)
        for i in range(diff_days + 1):
            thisday = day0 + datetime.timedelta(days=i)
            self._thisday = thisday
            initial_file_dir = self.data_root_dir / self.site / thisday.strftime('%Y')

            file_patterns = []
            if self.data_file_type == 'eiscat-hdf5':
                file_patterns.append('EISCAT')
            elif self.data_file_type == 'madrigal-hdf5':
                file_patterns.append('MAD6400')
            elif self.data_file_type == 'eiscat-mat':
                pass
            file_patterns.append(thisday.strftime('%Y-%m-%d'))
            file_patterns.append(self.modulation)
            file_patterns.append(self.antenna.lower())

            # remove empty str
            file_patterns = [pattern for pattern in file_patterns if str(pattern)]

            search_pattern = '*' + '*'.join(file_patterns) + '*'
            if self.data_file_type == 'eiscat-mat':
                search_pattern = search_pattern + '/'
            recursive = self.data_search_recursive
            done = super().search_data_files(
                initial_file_dir=initial_file_dir, search_pattern=search_pattern, recursive=recursive
            )

            # Validate file paths

            if not done and self.download:
                done = self.download_data()
                if done:
                    done = super().search_data_files(
                        initial_file_dir=initial_file_dir, search_pattern=search_pattern, recursive=recursive
                    )
                else:
                    print('Cannot find files from the online database!')

        return done

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
        if value is None:
            self._site = None
            return
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












