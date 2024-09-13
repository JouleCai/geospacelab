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
    def __init__(self, file_path, file_type='cdf', variable_name_dict=None, direct_load=True, dt_fr=None, dt_to=None, **kwargs):

        self.file_path = pathlib.Path(file_path)
        self.file_type = file_type
        self.variables = {}
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

    def load_cdf_data(self,):
        """
        load the data from the cdf file
        :return:
        """
        cdf_file = cdflib.CDF(self.file_path)
        cdf_info = cdf_file.cdf_info()
        variables = cdf_info.zVariables

        if dict(self.variable_name_dict):
            new_dict = {vn: vn for vn in variables}
            self.variable_name_dict = pybasic.dict_set_default(self.variable_name_dict, **new_dict)

        epochs = cdf_file.varget(variable=self.variable_name_dict['CDF_EPOCH'])
        if self.dt_fr is not None:
            t = self.dt_fr
            epoch_fr = cdflib.cdfepoch.compute_epoch([t.year, t.month, t.day, t.hour, t.minute, t.second])
        else:
            epoch_fr = epochs[0]
        if self.dt_to is not None:
            t = self.dt_to
            epoch_to = cdflib.cdfepoch.compute_epoch([t.year, t.month, t.day, t.hour, t.minute, t.second])
        else:
            epoch_to = epochs[-1]
        ind_t = np.where((epochs >= epoch_fr) & (epochs <= epoch_to))[0]

        num_data = len(epochs)
        for var_name, cdf_var_name in self.variable_name_dict.items():
            if var_name == 'CDF_EPOCH':
                epochs = epochs[ind_t]
                epochs = cdflib.cdfepoch.unixtime(epochs)
                dts = [datetime.timedelta(seconds=epoch) + datetime.datetime(1970, 1, 1, 0, 0, 0) for epoch in epochs]
                self.variables['SC_DATETIME'] = np.array(dts).reshape((len(dts), 1))
                continue
            # if var_name == 'CDF_EPOCH':
            #     epochs = epochs[ind_t]
            #     dts = cdflib.cdfepoch.to_datetime(epochs)
            #     self.variables['SC_DATETIME'] = np.array(dts).reshape((len(dts), 1))
            #     continue
            var = cdf_file.varget(variable=cdf_var_name)
            var = np.array(var)
            vshape = var.shape

            if num_data not in vshape:
                self.variables[var_name] = var
                continue

            if len(vshape) == 1:
                var = var[:, np.newaxis]
            self.variables[var_name] = var[ind_t, ::]


