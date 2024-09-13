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
    'JN_e': 'EL_I_FLUX',
    'JN_i': 'ION_I_FLUX',
    'JE_e': 'EL_I_ENER',
    'JE_i': 'ION_I_ENER',
    'E_e_MEAN': 'EL_M_ENER',
    'E_i_MEAN': 'ION_M_ENER',
    'jE_e': 'EL_D_ENER',
    'jE_i': 'ION_D_ENER',
    'jN_e': 'EL_D_FLUX',
    'jN_i': 'ION_D_FLUX',
    'SC_GEO_LAT': 'GDLAT',
    'SC_GEO_LON': 'GLON',
    'SC_GEO_ALT': 'GDALT',
    'SC_MAG_LAT': 'MLAT',
    'SC_MAG_LON': 'MLONG',
    'SC_MAG_MLT': 'MLT',
    'CH_CTRL_E':  'CH_CTRL_ENER',
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
            data_parameters = list(zip(*tuple(fh5['Data']['Array Layout']['1D Parameters']['Data Parameters'][:])))
            data = fh5['Data']['Array Layout']['1D Parameters']
            var_names_h5 = [vn.decode('UTF-8') for vn in data_parameters[0]]
            nvar_h5 = len(var_names_h5)
            vars_h5 = {}
            nrow = data['gdlat'][:].shape[0]
            for ip in range(nvar_h5):
                vars_h5[var_names_h5[ip]] = np.array(data[var_names_h5[ip].lower()]).reshape((nrow, 1))

            data_parameters = list(zip(*tuple(fh5['Data']['Array Layout']['2D Parameters']['Data Parameters'][:])))
            data = fh5['Data']['Array Layout']['2D Parameters']
            var_names_h5 = [vn.decode('UTF-8') for vn in data_parameters[0]]
            nvar_h5 = len(var_names_h5)
            for ip in range(nvar_h5):
                vars_h5[var_names_h5[ip]] = np.array(data[var_names_h5[ip].lower()]).T

            for var_name, var_name_h5 in self.variable_name_dict.items():
                try:
                    self.variables[var_name] = vars_h5[var_name_h5]
                except KeyError:
                    print(f"{var_name} is None!")
                    self.variables[var_name] = None

            # channel grid
            energy_spacing = vars_h5['CH_CTRL_ENER'] / 2
            E_chs = fh5['Data']['Array Layout']['ch_energy'][:]
            E_chs = np.array(E_chs).reshape(1, E_chs.shape[0])
            newchannel = E_chs - energy_spacing
            boudarychannel = E_chs[0, -1] + energy_spacing[:, -1]
            boudarychannel = boudarychannel.reshape((boudarychannel.shape[0], 1))
            self.variables['ENERGY_CHANNEL_GRID'] = np.hstack((newchannel, boudarychannel))

            # add datetime
            timestamps = fh5['Data']['Array Layout']['timestamps'][:]
            dt0 = datetime.datetime(1970, 1, 1, 0, 0, 0)

            dts = np.array([dt0 + datetime.timedelta(seconds=secs)
                               for ind, secs in enumerate(timestamps)]).reshape(nrow, 1)
            self.variables['SC_DATETIME'] = dts

            self.variables['JN_e'][self.variables['JN_e'] < 1.] = np.nan
            self.variables['JN_i'][self.variables['JN_i'] < 1.] = np.nan
            self.variables['JE_e'][self.variables['JE_e'] < 1.] = np.nan
            self.variables['JE_i'][self.variables['JE_i'] < 1.] = np.nan
            self.variables['jN_e'][self.variables['jN_e'] < 1.] = np.nan
            self.variables['jN_i'][self.variables['jN_i'] < 1.] = np.nan
            self.variables['jE_e'][self.variables['jE_e'] < 1.] = np.nan
            self.variables['jE_i'][self.variables['jE_i'] < 1.] = np.nan