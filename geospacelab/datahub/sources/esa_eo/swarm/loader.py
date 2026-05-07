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
import geospacelab.toolbox.utilities.pylogging as mylog

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
        self, file_path=None, file_type='cdf', 
        product_version=None,
        variable_name_dict=None, 
        from_VirES = False,
        from_HAPI = False,
        from_FAST = False,
        sat_id=None,
        direct_load=True, dt_fr=None, dt_to=None, **kwargs):

        self.file_path = pathlib.Path(file_path) if file_path is not None else None
        self.file_type = file_type
        self.variables = {}
        self.product_version = product_version
        self.dt_fr = dt_fr
        self.dt_to = dt_to
        self.sat_id = sat_id.upper() if sat_id is not None else None
        
        self.from_HAPI = from_HAPI
        self.from_VirES = from_VirES
        self.from_FAST = from_FAST

        if variable_name_dict is None:
            variable_name_dict = default_variable_name_dict
        self.variable_name_dict = variable_name_dict

        if direct_load:
            self.load_data(**kwargs)

    def load_data(self, **kwargs):
        
        if self.from_VirES:
            kwargs_VirES = kwargs.pop('kwargs_VirES', {})
            self.load_from_VirES(**kwargs_VirES)
        elif self.from_HAPI:
            kwargs_HAPI = kwargs.pop('kwargs_HAPI', {})
            self.load_from_HAPI(**kwargs_HAPI)
        else:
            if 'cdf' in self.file_type.lower():
                self.load_cdf_data()
            else:
                raise NotImplementedError(f"File type {self.file_type} not supported for loading.")

    def load_cdf_data(self, var_names_cdf_epoch=None, var_names_independent_time=None):
        """
        load the data from the cdf file
        :return:
        """
        if var_names_independent_time is None:
            var_names_independent_time = []
        if var_names_cdf_epoch is None:
            var_names_cdf_epoch = [vn_cdf for vn, vn_cdf in self.variable_name_dict.items() if 'CDF_EPOCH' in vn]
        
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
                shape = epochs.shape
                epochs_ = epochs.flatten()
                inds_epoch = np.where((epochs_>0) & (np.isfinite(epochs_)))[0]
                if epochs_.size != inds_epoch.size:
                    dts = np.full_like(epochs_, None, dtype=object)
                    epochs_valid = np.array(cdflib.cdfepoch.unixtime(epochs_[inds_epoch]))
                    dts_valid = [datetime.timedelta(seconds=epoch) + datetime.datetime(1970, 1, 1, 0, 0, 0) for epoch in epochs_valid]
                    dts[inds_epoch] = dts_valid
                    variables_cdf['DATETIME_' + vn_cdf] = np.array(dts).reshape(shape)
                    epoch_lengths.append(shape[0])
                else:
                    epochs = np.array(cdflib.cdfepoch.unixtime(epochs.flatten()))
                    dts = [datetime.timedelta(seconds=epoch) + datetime.datetime(1970, 1, 1, 0, 0, 0) for epoch in epochs.flatten()]
                    variables_cdf['DATETIME_' + vn_cdf] = np.array(dts).reshape(shape)
                    epoch_lengths.append(shape[0])
            
            variables_cdf[vn_cdf] = np.array(cdf_file.varget(variable=vn_cdf))
        
        variables = {}
        for var_name, var_name_cdf in self.variable_name_dict.items():
            if var_name_cdf not in variables_cdf:
                mylog.StreamLogger.warning(f"Variable name {var_name_cdf} not found in the cdf file.")
                continue
            data = variables_cdf[var_name_cdf]
            if len(data.shape) == 0:
                data = np.array([data])
            else:
                if data.shape[0] in epoch_lengths and var_name not in var_names_independent_time:
                    if len(data.shape) == 1:
                        data = data[:, np.newaxis]
            if 'CDF_EPOCH' in var_name:
                var_name_t = var_name.replace('CDF_EPOCH', 'SC_DATETIME')
                data_t = variables_cdf['DATETIME_' + var_name_cdf]
                if len(data_t.shape) == 1:
                    data_t = data_t[:, np.newaxis]
                variables[var_name_t] = data_t
            variables[var_name] = data
        
        self.variables = variables
        
    def load_from_VirES(self, collection=None, kwargs_products=None):
        from viresclient import SwarmRequest
        mylog.StreamLogger.info(f"Loading data from VirES for collection {collection} with products {kwargs_products}.")
        request = SwarmRequest()
        request.set_collection(collection)
        request.set_products(**kwargs_products)
        data = request.get_between(start_time=self.dt_fr, end_time=self.dt_to)
        mylog.StreamLogger.info(f"Data loaded from VirES.")
        return data
    
    def load_from_HAPI(
        self, 
        server="https://vires.services/hapi", 
        dataset=None, 
        parameters=""):
        from hapiclient import hapi
        mylog.StreamLogger.info(f"Loading data from HAPI for server {server}, dataset {dataset}, parameters {parameters}.")
        data, meta = hapi(server, dataset, parameters, self.dt_fr.isoformat(), self.dt_to.isoformat())
        mylog.StreamLogger.info(f"Data loaded from HAPI.")
        return data, meta
