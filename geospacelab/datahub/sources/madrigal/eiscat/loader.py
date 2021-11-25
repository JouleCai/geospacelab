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

from geospacelab import preferences as prf

import geospacelab.toolbox.utilities.pydatetime as dttool
import geospacelab.toolbox.utilities.pylogging as pylog
import geospacelab.toolbox.utilities.numpyarray as arraytool
import geospacelab.datahub.sources.madrigal.eiscat as eiscat
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
    def __init__(self, file_path, file_type="eiscat-hdf5"):
        self.variables = {}
        self.metadata = {}
        self.file_path = file_path
        self.file_type = file_type

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
                    elif nrec_group == 'par1d':
                        num_gates = int(np.max(nrec))
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

            vars['DATETIME'] = vars['DATETIME_1'] + (vars['DATETIME_2'] - vars['DATETIME_2']) / 2

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

        # check height and range
        vars['HEIGHT'] = vars['HEIGHT'] / 1000.
        vars['RANGE'] = vars['RANGE'] / 1000.
        inds = np.where(np.isnan(vars['HEIGHT']))
        for i in range(len(inds[0])):
            ind_0 = inds[0][i]
            ind_1 = inds[1][i]
            x0 = np.arange(vars['HEIGHT'].shape[0])
            y0 = vars['HEIGHT'][:, ind_1]
            xp = x0[np.where(np.isfinite(y0))[0]]
            yp = y0[np.isfinite(y0)]
            vars['HEIGHT'][ind_0, ind_1] = np.interp(x0[ind_0], xp, yp)

            x0 = np.arange(vars['RANGE'].shape[0])
            y0 = vars['RANGE'][:, ind_1]
            xp = x0[np.isfinite(y0)]
            yp = y0[np.isfinite(y0)]
            vars['RANGE'][ind_0, ind_1] = np.interp(x0[ind_0], xp, yp)

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
        raise NotImplemented


if __name__ == "__main__":
    dir_root = prf.datahub_data_root_dir

    fp = pathlib.Path('examples') / "EISCAT_2005-09-01_steffe_64@32m.hdf5"

    Loader(file_path=fp)
