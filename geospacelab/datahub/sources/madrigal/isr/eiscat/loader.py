# Licensed under the BSD 3-Clause License
# Copyright (C) 2021 GeospaceLab (geospacelab)
# Author: Lei Cai, Space Physics and Astronomy, University of Oulu

__author__ = "Lei Cai"
__copyright__ = "Copyright 2021, GeospaceLab"
__license__ = "BSD-3-Clause License"
__email__ = "lei.cai@oulu.fi"
__docformat__ = "reStructureText"


import h5py
import numpy as np
import pathlib
import re
import scipy.interpolate as si

from geospacelab.config import pref as prf

import geospacelab.toolbox.utilities.pydatetime as dttool
import geospacelab.datahub.sources.madrigal.isr.eiscat as eiscat
import geospacelab.toolbox.utilities.pylogging as mylog


default_variable_names = [
    'DATETIME', 'DATETIME_1', 'DATETIME_2',
    'MAGIC_CONSTANT', 'r_SCangle', 'r_m0_1', 'r_m0_2'
    'AZ', 'EL', 'P_Tx', 'HEIGHT', 'RANGE',
    'n_e', 'T_i', 'T_e', 'nu_i', 'v_i_los', 'comp_mix', 'comp_O_p',
    'n_e_err', 'T_i_err', 'T_e_err', 'nu_i_err', 'v_i_los_err', 'comp_mix_err', 'comp_O_p_err',
    'STATUS', 'RESIDUAL'
]


class Loader:
    def __init__(self, file_path, file_type="eiscat-hdf5", gate_num=None):
        self.variables = {}
        self.metadata = {}
        self.file_path = file_path
        self.file_type = file_type
        self.gate_num = gate_num
        self.load_data()

    def load_data(self):
        if self.file_type == 'eiscat-hdf5':
            self.load_eiscat_hdf5()
        elif self.file_type == 'eiscat-mat':
            self.load_eiscat_mat()
        elif self.file_type == 'madrigal-hdf5':
            self.load_madrigal_hdf5()
        else:
            raise ValueError

    def load_eiscat_hdf5(self):
        """
        Load EISCAT hdf5 file.
        :param file_paths: a list of the file fullpaths for the hdf5 files
        :return:
            load_obj: instance of Loader
                .variables: queried variables
                .metadata: metadata
        """

        var_name_dict = {
            'MAGIC_CONSTANT': 'Magic_const',
            'r_SCangle': 'SCangle',
            'r_m0_2': 'm02',
            'r_m0_1': 'm01',
            'AZ': 'az',
            'EL': 'el',
            'P_Tx': 'Pt',
            'T_SYS_1': 'Tsys1',
            'T_SYS_2': 'Tsys2',
            'HEIGHT': 'h',
            'RANGE': 'range',
            'n_e': 'Ne',
            'T_i': 'Ti',
            'T_r': 'Tr',
            'nu_i': 'Collf',
            'v_i_los': 'Vi',
            'comp_mix': 'pm',
            'comp_O_p': 'po+',
            'n_e_var': 'var_Ne',
            'T_i_var': 'var_Ti',
            'T_r_var': 'var_Tr',
            'nu_i_var': 'var_Collf',
            'v_i_los_var': 'var_Vi',
            'comp_mix_var': 'var_pm',
            'comp_O_p_var': 'var_po+',
            'STATUS': 'status',
            'RESIDUAL': 'res1'
        }

        site_info = {
            'T': 'UHF',
            'V': 'VHF',
            'K': 'KIR',
            'S': 'SOD',
            'L': 'ESR'
        }

        vars = {}
        metadata = {}

        with h5py.File(self.file_path, 'r') as fh5:
            var_info_list = eiscat.list_eiscat_hdf5_variables(fh5)
            h5_data = fh5['data']
            h5_metadata = fh5['metadata']
            ind_nrec = var_info_list['name'].index('nrec')
            nrec_group = var_info_list['group'][ind_nrec]
            nrec_group_ind = var_info_list['index'][ind_nrec]
            nrec = h5_data[nrec_group][nrec_group_ind]
            num_row = h5_data['utime'][0].shape[0]
            for var_name, var_name_h5 in var_name_dict.items():
                try:
                    ind_v = var_info_list['name'].index(var_name_h5)
                except ValueError:
                    mylog.StreamLogger.warning(f"'{var_name_h5}' is not in the hdf5 file!")
                    vars[var_name] = None
                    continue
                var_group = var_info_list['group'][ind_v]
                var_ind = var_info_list['index'][ind_v]
                var = h5_data[var_group][var_ind]
                if var_group == 'par0d':
                    vars[var_name] = var[0]
                elif var_group == 'par1d':
                    var = var.reshape(num_row, 1)
                    vars[var_name] = var
                elif var_group == 'par2d':
                    if nrec_group == 'par0d':
                        num_col = int(var.shape[0] / num_row)
                        if num_col != nrec[0]:
                            print("Note: the number of range gates doesn't match nrec!")
                        var = var.reshape(num_row, num_col)
                        if self.gate_num is None:
                            num_gates=self.gate_num = num_col
                        else:
                            num_gates = self.gate_num
                        var_array = np.empty((num_row, num_gates))
                        var_array[::] = np.nan
                        for i in range(num_row):
                            var_array[i, 0:num_col] = var[i, :]
                        var = var_array
                    elif nrec_group == 'par1d':
                        if self.gate_num is None:
                            self.gate_num = num_gates = int(np.max(nrec))
                        else:
                            num_gates = self.gate_num
                        var_array = np.empty((num_row, num_gates))
                        var_array[:, :] = np.nan
                        rec_ind_1 = 0
                        for i in range(num_row):
                            rec_ind_2 = int(rec_ind_1 + nrec[i])
                            var_array[i, :int(nrec[i])] = var[rec_ind_1:rec_ind_2]
                            rec_ind_1 = rec_ind_2
                        var = var_array
                    vars[var_name] = var

            # unix time to datetime
            utime1 = h5_data['utime'][0]
            dt1 = dttool.convert_unix_time_to_datetime_cftime(utime1)
            var = dt1.reshape(num_row, 1)
            var_name = 'DATETIME_1'
            vars[var_name] = var

            utime2 = h5_data['utime'][1]
            dt2 = dttool.convert_unix_time_to_datetime_cftime(utime2)
            var = dt2.reshape(num_row, 1)
            var_name = 'DATETIME_2'
            vars[var_name] = var

            vars['P_Tx'] = vars['P_Tx'] / 1e3   # in kW

            metadata['r_XMITloc'] = [h5_data['par0d'][2][0], h5_data['par0d'][3][0], h5_data['par0d'][4][0]]
            metadata['r_RECloc'] = [h5_data['par0d'][5][0], h5_data['par0d'][6][0], h5_data['par0d'][7][0]]
            metadata['site_name'] = site_info[h5_metadata['names'][1][1].decode('UTF-8').strip()]
            metadata['pulse_code'] = h5_metadata['names'][0][1].decode('UTF-8').strip()
            metadata['antenna'] = h5_metadata['names'][2][1].decode('UTF-8').strip()
            metadata['GUISDAP_version'] = h5_metadata['software']['GUISDAP_ver'][0, 0].decode('UTF-8').strip()
            metadata['rawdata_path'] = ''
            metadata['scan_mode'] = ''
            metadata['affiliation'] = ''
            try:
                title = h5_metadata['schemes']['DataCite']['Title'][0][0].decode('UTF-8').strip()
                rc = re.compile(r'\d{4}-\d{2}-\d{2}_([\S]+)@')
                exp_id = rc.findall(title)[0]
                metadata['modulation'] = exp_id.split('_')[-1]
                metadata['pulse_code'] = exp_id.replace('_'+metadata['modulation'], '')
            except KeyError:
                mylog.StreamLogger.warning("'Title is not listed in the metadata!'")

            try:
                # 'gfd' not in list before 2001?
                metadata['rawdata_path'] = h5_metadata['software']['gfd']['data_path'][0, 0].decode('UTF-8').strip()
                metadata['scan_mode'] = metadata['rawdata_path'].split('_')[1]
                metadata['affiliation'] = metadata['rawdata_path'].split('@')[0].split('_')[-1]
            except KeyError:
                mylog.StreamLogger.warning(
                    "'gfd' is not listed in the metadata! Affect 'rawdata_path', 'scan_mode', and 'affiliation'."
                )

        vars_add = {}
        for var_name, value in vars.items():
            if '_var' in var_name:
                var_name_err = var_name.replace('_var', '_err')
                vars_add[var_name_err] = np.sqrt(vars[var_name])
        vars.update(vars_add)

        vars['DATETIME'] = vars['DATETIME_1'] + (vars['DATETIME_2'] - vars['DATETIME_1'])/2
        vars['T_e'] = vars['T_i'] * vars['T_r']
        vars['T_e_err'] = vars['T_e'] * np.sqrt((vars['T_i_err']/vars['T_i'])**2
                                                + (vars['T_r_err']/vars['T_r'])**2)
        vars['AZ'] = vars['AZ'] % 360.
        # check height and range
        vars['HEIGHT'] = vars['HEIGHT'] / 1000.
        vars['RANGE'] = vars['RANGE'] / 1000.

        inds_nan = np.where(np.isnan(vars['HEIGHT']))
        if list(inds_nan):
            m, n = vars['HEIGHT'].shape
            for i in range(m):
                yy = vars['HEIGHT'][i, :].flatten()
                iii = np.where(~np.isfinite(yy))
                if not list(iii):
                    continue
                iii = np.where(np.isfinite(yy))[0]
                xx = np.arange(0, n)
                f = si.interp1d(xx[iii], yy[iii], kind='linear', bounds_error=False, fill_value='extrapolate')
                yy_new = f(xx)
                vars['HEIGHT'][i, :] = yy_new

                yy = vars['RANGE'][i, :].flatten()
                iii = np.where(~np.isfinite(yy))
                if not list(iii):
                    continue
                iii = np.where(np.isfinite(yy))[0]
                xx = np.arange(0, n)
                f = si.interp1d(xx[iii], yy[iii], kind='linear', bounds_error=False, fill_value='extrapolate')
                yy_new = f(xx)
                vars['RANGE'][i, :] = yy_new

        if np.isscalar(vars['AZ']):
            az = np.empty((vars['DATETIME_1'].shape[0], 1))
            az[:, :] = vars['AZ']
            vars['AZ'] = az
        if np.isscalar(vars['EL']):
            el = np.empty((vars['DATETIME_1'].shape[0], 1))
            el[:, :] = vars['EL']
            vars['EL'] = el
        vars['AZ'] = np.mod(vars['AZ'], 360)
        vars['EL'] = np.mod(vars['EL'], 360)

        self.variables = vars
        self.metadata = metadata

    def load_eiscat_mat(self):

        raise NotImplemented

    def load_madrigal_hdf5(self):
        from scipy.signal import argrelmin, argrelmax

        var_name_dict = {
            # 'MAGIC_CONSTANT': 'Magic_const',
            'r_SCangle': 'HSA',
            # 'r_m0_2': 'm02',
            # 'r_m0_1': 'm01',
            'AZ': 'AZM',
            'EL': 'ELM',
            'P_Tx': 'POWER',
            'T_SYS_1': 'SYSTMP',
            'T_SYS_2': 'SYSTMP',
            'HEIGHT': 'GDALT',
            'RANGE': 'RANGE',
            'n_e': 'NE',
            'T_i': 'TI',
            'T_r': 'TR',
            'nu_i': 'CO',
            'v_i_los': 'VO',
            'comp_mix': 'PM',
            'comp_O_p': 'PO+',
            'n_e_err': 'DNE',
            'T_i_err': 'DTI',
            'T_r_err': 'DTR',
            'nu_i_err': 'DCO',
            'v_i_los_err': 'DVO',
            'comp_mix_err': 'DPM',
            'comp_O_p_err': 'DPO+',
            'STATUS': 'GFIT',
            'RESIDUAL': 'CHISQ'
        }
        with h5py.File(self.file_path, 'r') as fh5:

            # load metadata
            metadata = {}

            exp_params = fh5['Metadata']['Experiment Parameters'][:]
            exp_params = list(zip(*tuple(exp_params)))
            fn_id = exp_params[0].index(b'Cedar file name')
            fn = exp_params[1][fn_id].decode('UTF-8')
            rc = re.compile(r'^MAD[\w]+_[\d]{4}-[\d]{2}-[\d]{2}_([\w.]+)@(\w+)\.hdf5')
            rm = rc.findall(fn)[0]
            pattern_1 = rm[0]
            pattern_2 = rm[1]
            metadata['modulation'] = pattern_1.split('_')[-1]
            metadata['pulse_code'] = pattern_1.replace('_' + metadata['modulation'], '')

            antenna = pattern_2
            if 'uhf' in antenna:
                sitename = 'UHF'
            elif 'vhf' in antenna:
                sitename = 'VHF'
            elif 'sod' in antenna:
                sitename = 'SOD'
            elif 'kir' in antenna:
                sitename = 'KIR'
            elif '32' in antenna or '42' in antenna:
                sitename = 'ESR'
            else:
                print(antenna)
                raise AttributeError
            metadata['site_name'] = sitename

            lat_id = exp_params[0].index(b'instrument latitude')
            lat = float(exp_params[1][lat_id].decode('UTF-8'))
            lon_id = exp_params[0].index(b'instrument longitude')
            lon = float(exp_params[1][lon_id].decode('UTF-8'))
            alt_id = exp_params[0].index(b'instrument altitude')
            alt = float(exp_params[1][alt_id].decode('UTF-8')) * 1000

            metadata['r_RECloc'] = [lat, lon, alt]
            if sitename in ['ESR', 'UHF', 'VHF', 'TRO']:
                metadata['r_XMITloc'] = [lat, lon, alt]
            else:
                metadata['r_XMITloc'] = [69.583, 19.21, 30]
            metadata['antenna'] = antenna
            metadata['GUISDAP_version'] = ''
            metadata['rawdata_path'] = ''
            metadata['scan_mode'] = ''
            metadata['affiliation'] = ''
            metadata['rawdata_path'] = ''
            metadata['scan_mode'] = ''
            metadata['affiliation'] = ''

            # load data
            data = fh5['Data']['Table Layout'][:]
            data = list(zip(*tuple(data)))
            data_parameters = list(zip(*tuple(fh5['Metadata']['Data Parameters'][:])))
            var_names_h5 = [vn.decode('UTF-8') for vn in data_parameters[0]]
            nvar_h5 = len(var_names_h5)
            ran_id = var_names_h5.index('RANGE')
            ran = data[ran_id]
            inds_ran_min = argrelmin(np.array(ran))[0]
            inds_ran_min = np.append(inds_ran_min, 0)
            inds_ran_min.sort()

            inds_ran_max = argrelmax(np.array(ran))[0]
            inds_ran_max = np.append(inds_ran_max, len(ran)-1)
            inds_ran_max.sort()

            if self.gate_num is None:
                self.gate_num = ngates_max = np.max(np.diff(inds_ran_max))
            else:
                ngates_max = self.gate_num
            num_row = inds_ran_min.shape[0]
            data_array = np.empty((nvar_h5, num_row, ngates_max))
            data_array[::] = np.nan
            for ip in range(nvar_h5):
                var_tmp = np.array(data[ip])
                for i in range(inds_ran_min.shape[0]):
                    ind1 = inds_ran_min[i]
                    ind2 = inds_ran_max[i]
                    data_array[ip, i, 0: ind2-ind1+1] = var_tmp[ind1: ind2+1]

            vars_h5 = {}
            for ip in range(nvar_h5):

                var = data_array[ip]
                test_unique = np.unique(var[0][~np.isnan(var[0])])
                if test_unique.shape[0] == 1 and not re.match(r'^D\w+', var_names_h5[ip]) and var_names_h5[ip] not in ['GFIT']:
                    # print(var_names_h5[ip])
                    var = np.array([list(set(var[i][~np.isnan(var[i])])) for i in range(num_row)]).reshape((num_row, 1))
                vars_h5[var_names_h5[ip]] = var

            vars = {}
            for var_name, var_name_h5 in var_name_dict.items():
                try:
                    vars[var_name] = vars_h5[var_name_h5]
                except KeyError:
                    vars[var_name] = None

            num_row = inds_ran_min.shape[0]
            # unix time to datetime
            utime1 = vars_h5['UT1_UNIX'].flatten()
            dt1 = dttool.convert_unix_time_to_datetime_cftime(utime1)
            var = dt1.reshape(num_row, 1)
            var_name = 'DATETIME_1'
            vars[var_name] = var

            utime2 = vars_h5['UT2_UNIX'].flatten()
            dt2 = dttool.convert_unix_time_to_datetime_cftime(utime2)
            var = dt2.reshape(num_row, 1)
            var_name = 'DATETIME_2'
            vars[var_name] = var

            vars['DATETIME'] = vars['DATETIME_1'] + (vars['DATETIME_2'] - vars['DATETIME_1']) / 2
            vars['T_e'] = vars['T_i'] * vars['T_r']
            vars['T_e_err'] = vars['T_e'] * np.sqrt((vars['T_i_err'] / vars['T_i']) ** 2
                                                    + (vars['T_r_err'] / vars['T_r']) ** 2)
            vars['AZ'] = vars['AZ'] % 360.

            inds_nan = np.where(np.isnan(vars['HEIGHT']))
            if list(inds_nan):
                m, n = vars['HEIGHT'].shape
                for i in range(m):
                    yy = vars['HEIGHT'][i, :].flatten()
                    iii = np.where(~np.isfinite(yy))
                    if not list(iii):
                        continue
                    iii = np.where(np.isfinite(yy))[0]
                    xx = np.arange(0, n)
                    f = si.interp1d(xx[iii], yy[iii], kind='linear', bounds_error=False, fill_value='extrapolate')
                    yy_new = f(xx)
                    vars['HEIGHT'][i, :] = yy_new

                    yy = vars['RANGE'][i, :].flatten()
                    iii = np.where(~np.isfinite(yy))
                    if not list(iii):
                        continue
                    iii = np.where(np.isfinite(yy))[0]
                    xx = np.arange(0, n)
                    f = si.interp1d(xx[iii], yy[iii], kind='linear', bounds_error=False, fill_value='extrapolate')
                    yy_new = f(xx)
                    vars['RANGE'][i, :] = yy_new


            self.variables = vars
            self.metadata = metadata
        # raise NotImplemented


if __name__ == "__main__":
    dir_root = prf.datahub_data_root_dir

    # fp = pathlib.Path('examples') / "EISCAT_2005-09-01_steffe_64@32m.hdf5"
    fp = pathlib.Path('examples') / "MAD6400_2021-03-10_beata_ant@uhfa.hdf5"
    Loader(file_path=fp, file_type = 'madrigal-hdf5')
