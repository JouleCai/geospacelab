# Licensed under the BSD 3-Clause License
# Copyright (C) 2021 GeospaceLab (geospacelab)
# Author: Lei Cai, Space Physics and Astronomy, University of Oulu

__author__ = "Lei Cai"
__copyright__ = "Copyright 2021, GeospaceLab"
__license__ = "BSD-3-Clause License"
__email__ = "lei.cai@oulu.fi"
__docformat__ = "reStructureText"


import netCDF4
import numpy as np
import cftime
import datetime


class Loader:
    def __init__(self, file_path, file_type='nc', load_data=True):
        self.file_path = file_path
        self.file_type = file_type
        self.variables = {}
        self.done = False
        if load_data:
            self.load()

    def load(self):
        fnc = netCDF4.Dataset(self.file_path)
        var_names_nc = fnc.variables.keys()
        variables = {}
        for var_name_nc in var_names_nc:
            if 'smr_' in var_name_nc:
                var_name = var_name_nc.upper()
            else:
                var_name = var_name_nc
            variables[var_name] = np.array(fnc[var_name_nc]) 
            if len(variables[var_name].shape) == 1:
                variables[var_name] = variables[var_name][:, np.newaxis]

        time_units = fnc['UNIX_TIME'].units

        variables['DATETIME'] = cftime.num2date(variables['UNIX_TIME'].flatten(),
                                                units=time_units,
                                                only_use_cftime_datetimes=False,
                                                only_use_python_datetimes=True)

        variables['DATETIME'] = np.reshape(variables['DATETIME'], (fnc['UNIX_TIME'].shape[0], 1))
        self.variables = variables
        self.done = True
        fnc.close()


