# Licensed under the BSD 3-Clause License
# Copyright (C) 2021 GeospaceLab (geospacelab)
# Author: Lei Cai, Space Physics and Astronomy, University of Oulu

import numpy as np
import datetime

import geospacelab.datahub as datahub
from geospacelab.datahub import DatabaseModel, FacilityModel, InstrumentModel, ProductModel
from geospacelab.datahub.sources.jhuapl import jhuapl_database
from geospacelab.config import prf
import geospacelab.toolbox.utilities.pybasic as basic
import geospacelab.toolbox.utilities.pylogging as mylog
import geospacelab.toolbox.utilities.pydatetime as dttool
from geospacelab.datahub.sources.jhuapl.ampere.grd.loader import Loader as default_Loader
from geospacelab.datahub.sources.jhuapl.ampere.grd.downloader import Downloader as default_Downloader
import geospacelab.datahub.sources.jhuapl.ampere.grd.variable_config as var_config


default_dataset_attrs = {
    'database': jhuapl_database,
    'facility': 'AMPERE',
    'product': 'GRD',
    'data_file_ext': 'nc',
    'data_root_dir': prf.datahub_data_root_dir / 'JHUAPL' / 'AMPERE' / 'GRD',
    'allow_load': True,
    'allow_download': True,
    'force_download': False,
    'data_search_recursive': False,
    'label_fields': ['database', 'facility', 'product'],
    'load_mode': 'AUTO',
    'time_clip': True,
}

default_variable_names = [
    'DATETIME', 
    'GRID_MLAT', 'GRID_MLT',
    'GRID_Jr'
    ]

# default_data_search_recursive = True

default_attrs_required = []


class Dataset(datahub.DatasetSourced):
    def __init__(self, **kwargs):
        kwargs = basic.dict_set_default(kwargs, **default_dataset_attrs)

        super().__init__(**kwargs)

        self.database = kwargs.pop('database', 'JHUAPL')
        self.facility = kwargs.pop('facility', 'AMPERE')
        self.instrument = kwargs.pop('instrument', 'Magnetometer')
        self.product = kwargs.pop('product', 'GRD')
        self.allow_download = kwargs.pop('allow_download', False)
        self.force_download = kwargs.pop('force_download', False)
        self.pole = kwargs.pop('pole', 'N')

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
            load_obj = self.loader(file_path, file_type=self.product.lower(), pole=self.pole)

            for var_name in self._variables.keys():
                value = load_obj.variables[var_name]
                self._variables[var_name].join(value)

            # self.select_beams(field_aligned=True)
        if self.time_clip:
            self.time_filter_by_range()

    def get_time_ind(self, ut):
        delta_sectime = [delta_t.total_seconds() for delta_t in (self['DATETIME'].value.flatten() - ut)]
        ind = np.where(np.abs(delta_sectime) == np.min(np.abs(delta_sectime)))[0][0]
        return ind

    def grid_fac(self, fac_data, mlat_data=None, mlt_data=None, mlt_res=0.05, mlat_res=0.05, interp_method='cubic'):
        import scipy.interpolate as si
        x = np.arange(0, 24, mlt_res)
        if self.pole == 'S':
            y = np.arange(-90, -40, mlat_res)
        else:
            y = np.arange(40, 90, mlat_res)

        grid_x, grid_y = np.meshgrid(x, y, indexing='ij')

        if mlt_data is None:
            mlt_data = self['GRID_MLT'].value[0, ::]
        if mlat_data is None:
            mlat_data = self['GRID_MLAT'].value[0, ::]

        xdata = np.vstack((mlt_data, mlt_data[0, :]))
        xdata[mlt_data.shape[0], :] = 24.

        ydata = np.vstack((mlat_data, mlat_data[0, :]))
        zdata = np.vstack((fac_data, fac_data[0, :]))

        grid_fac = si.griddata(
            (xdata.flatten(), ydata.flatten()),
            zdata.flatten(),
            (grid_x, grid_y),
            method=interp_method
        )
        grid_mlat = grid_y
        grid_mlt = grid_x
        return grid_mlat, grid_mlt, grid_fac

    def search_data_files(self, **kwargs):
        dt_fr = self.dt_fr
        dt_to = self.dt_to
        
        diff_days = dttool.get_diff_days(dt_fr, dt_to)
        dt0 = dttool.get_start_of_the_day(dt_fr)
        for i in range(diff_days + 1):
            thisday = dt0 + datetime.timedelta(days=i)
            
            initial_file_dir = kwargs.pop('initial_file_dir', None)
            if initial_file_dir is None:
                initial_file_dir = self.data_root_dir / thisday.strftime("%Y%m%d")
            for hh in range(24):
                start_hour = thisday + datetime.timedelta(hours=hh)
                end_hour = start_hour+datetime.timedelta(seconds=3600-1)
                if (start_hour < dt_fr) or (end_hour >dt_to):
                    continue 
                
                file_patterns = [
                    self.product.upper(),
                    start_hour.strftime("%Y%m%dT%H%M"),
                    end_hour.strftime("%Y%m%dT%H%M"),
                    self.pole
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
                    done = self.download_data(dt_fr=start_hour, dt_to=end_hour+datetime.timedelta(seconds=3600))
                    if done:
                        done = super().search_data_files(
                            initial_file_dir=initial_file_dir,
                            search_pattern=search_pattern,
                            allow_multiple_files=False
                        )
    
        return done
    
    def download_data(self, dt_fr=None, dt_to=None):
        if dt_fr is None:
            dt_fr = self.dt_fr
        if dt_to is None:
            dt_to = self.dt_to
        download_obj = self.downloader(
            dt_fr, dt_to,
            data_product=self.product.lower(),
            data_file_root_dir=self.data_root_dir, force_download=self.force_download, pole=self.pole)
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
