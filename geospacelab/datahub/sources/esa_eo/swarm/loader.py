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
    def __init__(self, file_path, file_type='cdf', variable_name_dict=None, direct_load=True, **kwargs):

        self.file_path = pathlib.Path(file_path)
        self.file_type = file_type
        self.variables = {}

        if variable_name_dict is None:
            variable_name_dict = default_variable_name_dict
        self.variable_name_dict = variable_name_dict

        if direct_load:
            self.load_data()

    def load_data(self, **kwargs):
        if self.file_type == 'cdf':
            self.load_cdf_data()

    def load_cdf_data(self):
        """
        load the data from the cdf file
        :return:
        """
        cdf_file = cdflib.CDF(self.file_path)
        cdf_info = cdf_file.cdf_info()
        variables = cdf_info['zVariables']

        if dict(self.variable_name_dict):
            new_dict = {vn: vn for vn in variables}
            self.variable_name_dict = pybasic.dict_set_default(self.variable_name_dict, **new_dict)

        for var_name, cdf_var_name in self.variable_name_dict.items():
            if var_name == 'CDF_EPOCH':
                epochs = cdf_file.varget(variable=cdf_var_name)
                epochs = cdflib.cdfepoch.unixtime(epochs)
                dts = [datetime.timedelta(seconds=epoch) + datetime.datetime(1970, 1, 1, 0, 0, 0) for epoch in epochs]
                self.variables['SC_DATETIME'] = np.array(dts).reshape((len(dts), 1))
                continue
            var = cdf_file.varget(variable=cdf_var_name)
            var = np.array(var)
            vshape = var.shape
            if len(vshape) == 1:
                var = var.reshape(vshape[0], 1)
            self.variables[var_name] = var


