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
import numpy as np
import cdflib

import geospacelab.toolbox.utilities.pybasic as pybasic

default_variable_name_dict = {
}


class LoaderModel(object):
    """
    :param file_path: the full path of the data file
    :type file_path: pathlib.Path or str
    :param file_type: the type of the file, [cdf]
    :param variable_name_dict: the dictionary for mapping the variable names from the cdf files to the dataset
    :type variable_name_dict: dict
    :param direct_load: call the method :meth:`~.LoadModel.load_data` directly or not
    :type direct_load: bool
    """
    def __init__(
        self, file_path, file_type='cdf', 
        product_version=None,
        variable_name_dict=None, direct_load=True, dt_fr=None, dt_to=None, **kwargs):

        self.file_path = pathlib.Path(file_path)
        self.file_type = file_type
        self.variables = {}
        self.product_version = product_version
        self.dt_fr = dt_fr
        self.dt_to = dt_to

        if variable_name_dict is None:
            variable_name_dict = default_variable_name_dict
        self.variable_name_dict = variable_name_dict

        if direct_load:
            self.load_data()

    def load_data(self, **kwargs):
        if self.file_type == 'cdf':
            self.load_cdf_data()

    def load_cdf_data(self, var_names_cdf_epoch=None, var_names_independent_time=None):
        """
        load the data from the cdf file
        :return:
        """
        if var_names_independent_time is None:
            var_names_independent_time = []
        
        cdf_file = cdflib.CDF(self.file_path)
        cdf_info = cdf_file.cdf_info()
        var_names_cdf = cdf_info.zVariables
        #  cdf_file.varattsget('MLT')
        # {'FIELDNAM': 'MLT', 'CATDESC': 'Magnetic local time.', 'Type': 'CDF_FLOAT', 'UNITS': 'hour', 'VAR_TYPE': 'data', 'DEPEND_0': 'Time', 'DISPLAY_TYPE': 'time_series', 'LABLAXIS': 'MLT', 'VALIDMIN': np.float32(0.0), 'VALIDMAX': np.float32(24.0)}
        # However, not all products have the full record of the data types and other attributibutes. For example, the AEJ_LPS product does not have the 'DEPEND_0' attribute for the variables that depend on time, and thus we need to specify the variable names of the variables that are epoch and independent of time.
        
        variables_cdf = {}
        epoch_lengths = []
        for vn_cdf in var_names_cdf:
            if vn_cdf in var_names_cdf_epoch:
                epochs = cdf_file.varget(variable=vn_cdf)
                epochs = np.array(cdflib.cdfepoch.unixtime(epochs))
                shape = epochs.shape
                dts = [datetime.timedelta(seconds=epoch) + datetime.datetime(1970, 1, 1, 0, 0, 0) for epoch in epochs.flatten()]
                variables_cdf[vn_cdf] = np.array(dts).reshape(shape)
                epoch_lengths.append(shape[0])
            else:
                variables_cdf[vn_cdf] = np.array(cdf_file.varget(variable=vn_cdf))
        
        variables = {}
        for var_name, var_name_cdf in self.variable_name_dict.items():
            if var_name_cdf not in variables_cdf:
                raise ValueError(f"Variable name {var_name_cdf} not found in the cdf file.")
            data = variables_cdf[var_name_cdf]
            if data.shape[0] in epoch_lengths and var_name not in var_names_independent_time:
                if len(data.shape) == 1:
                    data = data[:, np.newaxis]
                else:
                    pass
            if 'CDF_EPOCH' in var_name:
                var_name_ = var_name.replace('CDF_EPOCH', 'SC_DATETIME')
            else:
                var_name_ = var_name
            variables[var_name_] = data
        
        self.variables = variables

