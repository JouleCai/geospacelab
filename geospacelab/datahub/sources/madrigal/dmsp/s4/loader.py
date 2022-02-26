# Licensed under the BSD 3-Clause License
# Copyright (C) 2021 GeospaceLab (geospacelab)
# Author: Lei Cai, Space Physics and Astronomy, University of Oulu

__author__ = "Lei Cai"
__copyright__ = "Copyright 2021, GeospaceLab"
__license__ = "BSD-3-Clause License"
__email__ = "lei.cai@oulu.fi"
__docformat__ = "reStructureText"

import datetime
import pathlib
import h5py
import numpy as np

import geospacelab.toolbox.utilities.pylogging as mylog

# define the default variable name dictionary
default_variable_name_dict = {
    'T_i': 'TI',
    'T_e': 'TE',
    'COMP_O_p': 'PO+',
    'phi_E': 'ELEPOT',
    'SC_GEO_LAT': 'GDLAT',
    'SC_GEO_LON': 'GLON',
    'SC_GEO_ALT': 'GDALT',
    'SC_MAG_LAT': 'MLAT',
    'SC_MAG_LON': 'MLONG',
    'SC_MAG_MLT': 'MLT',
}


class Loader(object):
    """
    :param file_path: the full path of the data file
    :type file_path: pathlib.Path or str
    :param file_type: the type of the file, [cdf]
    :param variable_name_dict: the dictionary for mapping the variable names from the cdf files to the dataset
    :type variable_name_dict: dict
    :param direct_load: call the method :meth:`~.LoadModel.load_data` directly or not
    :type direct_load: bool
    """
    def __init__(self, file_path, file_type='hdf5', variable_name_dict=None, direct_load=True, **kwargs):

        self.file_path = pathlib.Path(file_path)
        self.file_ext = file_type
        self.variables = {}
        self.metadata = {}

        if variable_name_dict is None:
            variable_name_dict = default_variable_name_dict
        self.variable_name_dict = variable_name_dict

        if direct_load:
            self.load_data()

    def load_data(self, **kwargs):
        if self.file_ext == 'hdf5':
            self.load_hdf5_data()

    def load_hdf5_data(self):
        """
        load the data from the cdf file
        :return:
        """
        with h5py.File(self.file_path, 'r') as fh5:
            # load metadata
            metadata = {}


            # load data
            data = fh5['Data']['Table Layout'][:]

            data = list(zip(*tuple(data)))

            data_parameters = list(zip(*tuple(fh5['Metadata']['Data Parameters'][:])))
            var_names_h5 = [vn.decode('UTF-8') for vn in data_parameters[0]]
            nvar_h5 = len(var_names_h5)

            vars_h5 = {}
            nrow = len(data[0])
            for ip in range(nvar_h5):
                vars_h5[var_names_h5[ip]] = np.array(data[ip])

            for var_name, var_name_h5 in self.variable_name_dict.items():
                try:
                    self.variables[var_name] = vars_h5[var_name_h5].reshape((nrow, 1))
                except KeyError:
                    print(f"{var_name} is None!")
                    self.variables[var_name] = None

            # add datetime
            dts = [
                datetime.datetime(yy, mm, dd, HH, MM, SS)
                for yy, mm, dd, HH, MM, SS in zip(
                    vars_h5['YEAR'], vars_h5['MONTH'], vars_h5['DAY'],
                    vars_h5['HOUR'], vars_h5['MIN'], vars_h5['SEC']
                )
            ]
            self.variables['SC_DATETIME'] = np.array(dts).reshape((nrow, 1))

            self.metadata = metadata