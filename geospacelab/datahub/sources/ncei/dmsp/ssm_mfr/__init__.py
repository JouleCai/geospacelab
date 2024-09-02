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
from geospacelab.datahub.sources.ncei import ncei_database
from geospacelab.datahub.sources.ncei.dmsp import dmsp_facility
from geospacelab.config import prf
import geospacelab.toolbox.utilities.pybasic as basic
import geospacelab.toolbox.utilities.pylogging as mylog
import geospacelab.toolbox.utilities.pydatetime as dttool
# from geospacelab.datahub.sources.madrigal.satellites.dmsp.s1.loader import Loader as default_Loader
from geospacelab.datahub.sources.ncei.dmsp.ssm_mfr.downloader import Downloader as default_Downloader
# import geospacelab.datahub.sources.madrigal.satellites.dmsp.s1.variable_config as var_config
from geospacelab.datahub.sources.ncei.dmsp.ssm_mfr.loader import Loader as default_Loader
import geospacelab.datahub.sources.ncei.dmsp.ssm_mfr.variable_config as var_config 

default_dataset_attrs = {
    'database': ncei_database,
    'facility': dmsp_facility,
    'instrument': 'SSM',
    'product': 'SSM_MFR',
    'data_file_ext': '.MFR',
    'data_root_dir': prf.datahub_data_root_dir / 'NCEI' / 'DMSP' / 'SSM_MFR',
    'allow_load': True,
    'allow_download': True,
    'force_download': False,
    'data_search_recursive': False,
    'label_fields': ['database', 'facility', 'instrument', 'product'],
    'load_mode': 'AUTO',
    'time_clip': True,
    'add_AACGM': True,
    'add_APEX': False,
    'calib_orbit': False,
    'replace_orbit': True,
}

default_variable_names = [
    'SC_DATETIME',
    'SC_GEO_LAT',
    'SC_GEO_LON',
    'SC_GEO_ALT',
    'B_D', 'B_P', 'B_F', 'd_B_D', 'd_B_P', 'd_B_F'
    ]

# default_data_search_recursive = True

default_attrs_required = ['sat_id', ]


class Dataset(datahub.DatasetSourced):
    def __init__(self, **kwargs):
        kwargs = basic.dict_set_default(kwargs, **default_dataset_attrs)

        super().__init__(**kwargs)

        self.database = kwargs.pop('database', 'NCEI')
        self.facility = kwargs.pop('facility', 'DMSP')
        self.instrument = kwargs.pop('instrument', 'SSM')
        self.product = kwargs.pop('product', 'SSM_MFR')
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
            load_obj = self.loader(file_path)

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
        from scipy.interpolate import interp1d
        # check outliers
        orbit_obj = OrbitPosition_SSCWS(
            dt_fr=self.dt_fr - datetime.timedelta(minutes=30),
            dt_to=self.dt_to + datetime.timedelta(minutes=30),
            sat_id='dmsp' + self.sat_id.lower()
        )

        glat_1 = self['SC_GEO_LAT'].value.flatten()
        glon_1 = self['SC_GEO_LON'].value.flatten()
        if glat_1.size < 2:
            return

        dts_1 = self['SC_DATETIME'].value.flatten()
        dt0 = dttool.get_start_of_the_day(self.dt_fr)
        sectime_1 = [(dt - dt0).total_seconds() for dt in dts_1]

        glat_2 = orbit_obj['SC_GEO_LAT'].value.flatten()
        glon_2 = orbit_obj['SC_GEO_LON'].value.flatten()
        dts_2 = orbit_obj['SC_DATETIME'].value.flatten()
        sectime_2 = [(dt - dt0).total_seconds() for dt in dts_2]

        factor = np.pi / 180.
        sin_glon_1 = np.sin(glon_1 * factor)
        sin_glon_2 = np.sin(glon_2 * factor)
        cos_glon_2 = np.cos(glon_2 * factor)
        itpf_sin = interp1d(sectime_2, sin_glon_2, kind='cubic', bounds_error=False, fill_value='extrapolate')
        itpf_cos = interp1d(sectime_2, cos_glon_2, kind='cubic', bounds_error=False, fill_value='extrapolate')
        sin_glon_2_i = itpf_sin(sectime_1)
        sin_glon_2_i = np.where(sin_glon_2_i > 1., 1., sin_glon_2_i)
        sin_glon_2_i = np.where(sin_glon_2_i < -1., -1., sin_glon_2_i)
        
        cos_glon_2_i = itpf_cos(sectime_1)
        cos_glon_2_i = np.where(cos_glon_2_i > 1., 1., cos_glon_2_i)
        cos_glon_2_i = np.where(cos_glon_2_i < -1., -1., cos_glon_2_i)
        
        rad = np.sign(sin_glon_2_i) * (np.pi / 2 - np.arcsin(cos_glon_2_i))
        glon_new = rad / factor
        # rad = np.where((rad >= 0), rad, rad + 2 * numpy.pi)

        ind_outliers = np.where(np.abs(sin_glon_1 - sin_glon_2_i) > 0.03)[0]

        if self.replace_orbit:
            glon_1 = glon_new
        else:
            glon_1[ind_outliers] = glon_new[ind_outliers]
        self['SC_GEO_LON'].value = glon_1.reshape((glon_1.size, 1))

    def search_data_files(self, **kwargs):

        dt_fr = self.dt_fr
        dt_to = self.dt_to

        diff_days = dttool.get_diff_days(dt_fr, dt_to)

        dt0 = dttool.get_start_of_the_day(dt_fr)

        for i in range(diff_days + 1):
            this_day = dt0 + datetime.timedelta(days=i)

            initial_file_dir = kwargs.pop(
                'initial_file_dir', self.data_root_dir / self.sat_id.upper() / \
                                    this_day.strftime('%Y') / this_day.strftime('%m')
            )

            file_patterns = [
                'SSM',
                this_day.strftime('%Y%m%d'),
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

    def interp_evenly(self, time_res=1, time_res_o=1, dt_fr=None, dt_to=None):
        from scipy.interpolate import interp1d
        import geospacelab.toolbox.utilities.numpymath as nm

        ds_new = datahub.DatasetUser(dt_fr=self.dt_fr, dt_to=self.dt_to, visual=self.visual)
        ds_new.clone_variables(self)
        dts = ds_new['SC_DATETIME'].value.flatten()
        dt0 = dttool.get_start_of_the_day(dts[0])
        x_0 = np.array([(dt - dt0).total_seconds() for dt in dts])
        if dt_fr is None:
            dt_fr = dts[0] - datetime.timedelta(
                seconds=np.floor(((dts[0] - ds_new.dt_fr).total_seconds() / time_res)) * time_res
            )
            dt_fr = datetime.datetime(dt_fr.year, dt_fr.month, dt_fr.day, dt_fr.hour, dt_fr.minute, dt_fr.second)
        if dt_to is None:
            dt_to = dts[-1] + datetime.timedelta(
                seconds=np.floor(((ds_new.dt_to - dts[-1]).total_seconds() / time_res)) * time_res
            )
            dt_to = datetime.datetime(dt_to.year, dt_to.month, dt_to.day, dt_to.hour, dt_to.minute, dt_to.second)
        sec_fr = (dt_fr - dt0).total_seconds()
        sec_to = (dt_to - dt0).total_seconds()
        x_1 = np.arange(sec_fr, sec_to + time_res / 2, time_res)

        f = interp1d(x_0, x_0, kind='nearest', bounds_error=False, fill_value=(x_0[0], x_0[-1]))
        pseudo_x = f(x_1)
        mask = np.abs(pseudo_x-x_1) > time_res_o/1.5

        dts_new = np.array([dt0 + datetime.timedelta(seconds=sec) for sec in x_1])
        ds_new['SC_DATETIME'].value = dts_new.reshape((dts_new.size, 1))

        period_var_dict = {'SC_GEO_LON': 360.,
                           'SC_AACGM_LON': 360.,
                           'SC_APEX_LON': 360.,
                           'SC_GEO_LST': 24.,
                           'SC_AACGM_MLT': 24.,
                           'SC_APEX_MLT': 24}

        for var_name in ds_new.keys():
            if var_name in ['SC_DATETIME']:
                continue
            if var_name in period_var_dict.keys():
                var = ds_new[var_name].value.flatten()
                var_new = nm.interp_period_data(x_0, var, x_1, period=period_var_dict[var_name], method='linear', bounds_error=False)
                var_new[mask] = np.nan
                ds_new[var_name].value = var_new.reshape((dts_new.size, 1))
            else:
                method = 'linear' if 'FLAG' not in var_name else 'nearest'
                var = ds_new[var_name].value.flatten()
                f = interp1d(x_0, var, kind=method, bounds_error=False)
                var_new = f(x_1)
                var_new[mask] = np.nan
                ds_new[var_name].value = var_new.reshape((dts_new.size, 1))
        return ds_new

    def download_data(self, dt_fr=None, dt_to=None):
        if dt_fr is None:
            dt_fr = self.dt_fr
        if dt_to is None:
            dt_to = self.dt_to
        download_obj = self.downloader(
            dt_fr, dt_to,
            sat_id=self.sat_id,
            force_download=self.force_download
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
