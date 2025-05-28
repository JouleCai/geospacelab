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
import cftime
import re
import datetime

import geospacelab.toolbox.utilities.pylogging as mylog
import geospacelab.toolbox.utilities.pydatetime as dttool

pulse_code_dict = {
    'alternating': 97,
    'barker': 98,
    'single pulse': 115,
}

antenna_code_dict = {
    'zenith': 32,
    'misa': 31,
}

antenna_code_dict_r = {v: k for k, v in antenna_code_dict.items()}
pulse_code_dict_r = {v: k for k, v in pulse_code_dict.items()}

var_name_dict = {
    'AZ1': 'az1',
    'AZ2': 'az2',
    'EL1': 'el1',
    'EL2': 'el2',
    'PULSE_LENGTH': 'pl',
    'T_SYS': 'systmp',
    'POWER_NORM': 'pnrmd',
    'P_Tx': 'power',
    'MODE_TYPE': 'mdtyp',
    'POWER_LENGTH_F': 'pulf',
    'LAG_SPACING': 'dtau',
    'IPP': 'ipp',
    'f_Tx': 'tfreq',
    'v_PHASE_Tx': 'vtx',
    'v_PHASE_Tx_err': 'dvtx',
    'SCAN_TYPE': 'scntyp',
    'CYCN': 'cycn',
    'POSN': 'posn',
    'RANGE_RES': 'mresl',
    'RANGE': 'range',
    'SNR': 'snp3',
    'RESIDUAL': 'wchsq',
    'STATUS': 'gfit',
    'FIT_TYPE': 'fit_type',
    'FPI_QUALITY': 'fpi_dataqual',
    'ACF_NORM': 'fa',
    'ACF_NORM_ERR': 'dfa',
    'n_pp': 'popl',
    'n_pp_err': 'dpopl',
    'n_e': 'ne',
    'n_e_err': 'dne',
    'T_i': 'ti',
    'T_i_err': 'dti',
    'T_r': 'tr',
    'T_r_err': 'dtr',
    'nu_i': 'co',
    'nu_i_err': 'dco',
    'v_i_los': 'vo',
    'v_i_los_err': 'dvo',
    'comp_H_p': 'ph+',
    'comp_H_p_err': 'dph+',
    'comp_mix': 'pm',
    'comp_mix_err': 'dpm',
    'v_DOP_los': 'vdopp',
    'v_DOP_los_err': 'dvdopp',
    'HEIGHT': 'gdalt'
}


class Loader:
    """
    :param file_path: the file's full path
    :type file_path: pathlib.Path object
    :param file_type: the specific file type for the file being loaded. Options: ['TEC-MAT'], 'TEC-LOS', 'TEC-sites')
    :type file_type: str
    :param load_data: True, load without calling the method "load_data" separately.
    :type load_data: bool
    """
    def __init__(self, file_path, antenna='', pulse_code='', pulse_length=0, load_data=True):
        self.file_path = file_path
        self.antenna = antenna
        self.pulse_code = pulse_code
        self.pulse_length = pulse_length
        self.variables = {}
        self.metadata = {}

        self.done = False
        if load_data:
            self.load_from_table_layout()


    def load_from_table_layout(self):
        variables = {}

        with h5py.File(self.file_path, 'r') as fh5:
            table = fh5['Data']['Table Layout'][::]
            table_dtype = table.dtype

            # Get variable names
            table_var_names = list(table_dtype.fields.keys())
            # Get table data
            table_data = list(zip(*table))
            table_vars = {k: np.array(table_data[table_var_names.index(k)]) for k in table_var_names}
            # Check modes
            antenna_codes = table_vars['kinst'] = table_vars['kinst'].astype(np.int32)
            pulse_lengths = table_vars['pl'] = table_vars['pl'] * 1e6
            pulse_codes = table_vars['mdtyp'] = table_vars['mdtyp'].astype(np.int32)
            modes_ = list(zip(antenna_codes, pulse_codes, pulse_lengths))
            modes = np.empty(len(modes_), dtype=object)
            modes[:] = modes_
            modes_unique = self.get_modes(modes)

            mode = self.validate_mode(modes_unique)

            inds_mode = np.where((antenna_codes==mode[0]) & (pulse_codes==mode[1]) & (pulse_lengths==mode[2]) )[0]

            self.metadata['antenna'] = antenna_code_dict_r[mode[0]]
            self.metadata['pulse_code'] = pulse_code_dict_r[mode[1]]
            self.metadata['pulse_length'] = mode[2]

            # Grid data
            ut_unix_1 = table_vars['ut1_unix'][inds_mode]
            ut_unix_1_unique = np.unique(ut_unix_1)
            num_ut = len(ut_unix_1_unique)
            vars_fh5 = {}
            if len(ut_unix_1) % num_ut == 0:
                num_gate = int(len(ut_unix_1) / num_ut)

                for k, v in table_vars.items():
                    vars_fh5[k] = v[inds_mode].reshape((num_ut, num_gate))
            else:
                gate_inds = []
                gate_nums = []
                for t in ut_unix_1_unique:
                    ii = np.where(ut_unix_1==t)[0]
                    ran = table_vars['range'][inds_mode][ii]
                    diff_ran = np.diff(ran)
                    if any(diff_ran < 0): # duplicated ranges
                        iii = range(np.where(diff_ran<0)[0][0]+1)
                        ii = ii[iii]
                    gate_inds.append(ii)

                    gate_nums.append(len(gate_inds[-1]))
                max_gate_num = np.max(gate_nums)

                for k, v in table_vars.items():
                    vars_fh5[k] = np.empty((num_ut, max_gate_num))
                    vars_fh5[k][::] = np.nan
                    for i, inds in enumerate(gate_inds):
                        vars_fh5[k][i, 0:len(inds)] = v[inds_mode][inds]
            
            # Assign data
            records = fh5['Metadata']['_record_layout'][0]
            rec_var_names = np.array(list(records.dtype.fields.keys()))
            rec_vars = {str(rec_var_names[i]): int(records[i]) for i in range(len(rec_var_names))}
            for var_name, var_name_fh5 in var_name_dict.items():
                if var_name_fh5 not in vars_fh5.keys():
                    mylog.StreamLogger.warning(f"The requested variable {var_name_fh5} does not exist in the data file!")
                    variables[var_name] = None
                    continue
                if rec_vars[var_name_fh5] == 1:
                    variables[var_name] = vars_fh5[var_name_fh5][:, 0][:, np.newaxis]
                else:
                    variables[var_name] = vars_fh5[var_name_fh5]

            variables['comp_O_p'] = 1. - variables['comp_mix'] - variables['comp_H_p']
            variables['comp_O_p_err'] = np.sqrt(variables['comp_mix_err']**2 + variables['comp_H_p_err']**2)

            # need to be check when AZ close to 0.
            variables['AZ1'] = variables['AZ1'] % 360.
            variables['AZ2'] = variables['AZ2'] % 360.
            variables['AZ'] = (variables['AZ1'] + variables['AZ2']) / 2
            diff_az = np.abs(variables['AZ1'] - variables['AZ2'])
            variables['AZ'] = np.where(diff_az<180, variables['AZ'], ((variables['AZ1'] + variables['AZ2'] + 360) / 2) % 360)
            variables['EL'] = (variables['EL1'] + variables['EL2']) / 2 
            
            variables['DATETIME_1'] = dttool.convert_unix_time_to_datetime_cftime(vars_fh5['ut1_unix'][:, 0])[:, np.newaxis]
            variables['DATETIME_2'] = dttool.convert_unix_time_to_datetime_cftime(vars_fh5['ut2_unix'][:, 0])[:, np.newaxis] 
            variables['DATETIME'] = variables['DATETIME_1'] + (variables['DATETIME_2'] - variables['DATETIME_1']) / 2 
            variables['T_e'] = variables['T_i'] * variables['T_r']
            variables['T_e_err'] = variables['T_e'] * np.sqrt((variables['T_i_err'] / variables['T_i']) ** 2
                                                              + (variables['T_r_err'] / variables['T_r']) ** 2)
        self.variables = variables
         
        return

    def validate_mode(self, modes_unique):
        mode = None

        # Check antenna
        modes_antenna_matched = []
        try:
            for m in modes_unique:
                if m[0] == antenna_code_dict[self.antenna]:
                    modes_antenna_matched.append(m)
            if len(modes_antenna_matched) == 0:
                raise KeyError
            elif len(modes_antenna_matched) == 1:
                if str(self.pulse_code):
                    if pulse_code_dict[self.pulse_code] != modes_antenna_matched[0][1]:
                        raise ValueError
                if self.pulse_length > 0:
                    if self.pulse_length != modes_antenna_matched[0][2]:
                        raise ValueError
                return modes_antenna_matched[0]
            else:
                pass
        except Exception as e:
            if str(self.antenna):
                mylog.StreamLogger.error("Antenna {} is not found!".format(self.antenna.upper()))
            else:
                mylog.StreamLogger.error("Antenna must be specified!")
            self.list_modes()
            return None

        modes_pulse_code_matched = []
        try:
            for m in modes_antenna_matched:
                if m[1] == pulse_code_dict[self.pulse_code]:
                    modes_pulse_code_matched.append(m)
            if len(modes_pulse_code_matched) == 0:
                raise KeyError
            elif len(modes_pulse_code_matched) == 1:
                if self.pulse_length > 0:
                    if self.pulse_length != modes_antenna_matched[0][2]:
                        raise ValueError
                return modes_pulse_code_matched[0]
            else:
                pass
        except Exception as e:
            if str(self.pulse_code):
                mylog.StreamLogger.error("Pulse code {} is not found!".format(self.pulse_code.upper()))
            else:
                mylog.StreamLogger.error("Pulse code must be specified!")
            self.list_modes()
            return None

        modes_pulse_length_matched = []
        try:
            for m in modes_pulse_code_matched:
                if m[2] == self.pulse_length:
                    modes_pulse_length_matched.append(m)
            if len(modes_pulse_length_matched) == 0:
                raise ValueError
            elif len(modes_pulse_length_matched) == 1:
                return modes_pulse_length_matched[0]
            else:
                mylog.StreamLogger.error("Multiple modes found!")
                raise ValueError
        except Exception as e:
            if self.pulse_length > 0:
                mylog.StreamLogger.error("Pulse length {} is not found!".format(self.pulse_code.upper()))
            else:
                mylog.StreamLogger.error("Pulse length must be specified!")
            self.list_modes()
            return None
        return mode

    def list_modes(self):
        mylog.simpleinfo.info("List of the experiment modes:")

        for i, m in enumerate(self.metadata['MODES']):
            s = '{}, '.format(i)
            for k, v in m.items():
                s = s + "{}: {}, ".format(k, v)
            mylog.simpleinfo.info(s)
        return
    def get_modes(self, modes):

        modes_unique = np.unique(modes)

        self.metadata['MODES'] = []
        num_modes = self.metadata['NUM_MODES'] = len(modes_unique)
        for i in range(num_modes):
            self.metadata['MODES'].append({
                'ANTENNA_ID': modes_unique[i][0],
                'ANTENNA': antenna_code_dict_r[modes_unique[i][0]],
                'PULSE_CODE_ID': modes_unique[i][1],
                'PULSE_CODE': pulse_code_dict_r[modes_unique[i][1]],
                'PULSE_LENGTH': modes_unique[i][2]
            })
        return modes_unique

    def load(self):
        variables = {}
        metadata = {}
        with h5py.File(self.file_path, 'r') as fh5:
            data_fh5 = fh5['Data']
            array_layouts = list(data_fh5['Array Layout'].keys())

            antenna_codes = []
            pulse_codes = []
            pulse_lengths = []
            match = 0
            ind_array = None
            for i, array_layout_str in enumerate(array_layouts):
                res = re.search(r'kinst=([\d.]+).*mdtyp=([\d.]+).*pl=([\d.]+)', array_layout_str)
                antenna_code = int(float(res.groups()[0]))
                pulse_code = int(float(res.groups()[1]))
                pulse_length = float(res.groups()[2]) * 1e6
                antenna_codes.append(antenna_code)
                pulse_codes.append(pulse_code)
                pulse_lengths.append(pulse_length)
                if not str(self.antenna) or not str(self.pulse_code) or self.pulse_length == 0:
                    continue
                if antenna_code != antenna_code_dict[self.antenna] or \
                        pulse_code != pulse_code_dict[self.pulse_code] or \
                        pulse_length != self.pulse_length:
                    continue
                match = match + 1
                ind_array = i
            antenna_code_dict_r = {v: k for k, v in antenna_code_dict.items()}
            pulse_code_dict_r = {v: k for k, v in pulse_code_dict.items()}
            if match != 1:
                mylog.StreamLogger.error("The inputs do not match the Array Layouts! Check the Array Layout info below:")
                mylog.simpleinfo.info(
                        "{:10s}{:20s}{:30s}{:30s}".format(
                            'No.',
                            'antenna',
                            'pulse_code',
                            'pulse_length',
                        )
                    )
                for ind, (antenna_code, pulse_code, pulse_length) \
                        in enumerate(zip(antenna_codes, pulse_codes, pulse_lengths)):

                    mylog.simpleinfo.info(
                        "{:<10d}{:20s}{:30s}{:<30f}".format(
                            ind+1,
                            antenna_code_dict_r[antenna_code],
                            pulse_code_dict_r[pulse_code],
                            pulse_length,
                        )
                    )
                raise ValueError("Check the message above!")

            metadata['antenna'] = antenna_code_dict_r[antenna_codes[ind_array]]
            metadata['pulse_code'] = pulse_code_dict_r[pulse_codes[ind_array]]
            metadata['pulse_length'] = pulse_lengths[ind_array]

            vars_fh5 = {}
            array_layout_str = array_layouts[ind_array]
            fh5_vars_1d = data_fh5['Array Layout'][array_layout_str]['1D Parameters']
            for var_name in fh5_vars_1d.keys():
                if var_name == 'Data Parameters':
                    continue
                vars_fh5[var_name] = np.array(fh5_vars_1d[var_name])[:, np.newaxis]
            fh5_vars_2d = data_fh5['Array Layout'][array_layout_str]['2D Parameters']
            for var_name in fh5_vars_2d.keys():
                if var_name == 'Data Parameters':
                    continue
                vars_fh5[var_name] = np.array(fh5_vars_2d[var_name]).T
            vars_fh5['range'] = np.array(data_fh5['Array Layout'][array_layout_str]['range'])[np.newaxis, :]
            vars_fh5['timestamps'] = np.array(data_fh5['Array Layout'][array_layout_str]['timestamps'])[:, np.newaxis]
            for var_name, var_name_fh5 in var_name_dict.items():
                if var_name_fh5 not in vars_fh5.keys():
                    mylog.StreamLogger.warning(f"The requested variable {var_name_fh5} does not exist in the data file!")
                    variables[var_name] = None
                    continue
                variables[var_name] = vars_fh5[var_name_fh5]

            variables['comp_O_p'] = 1. - variables['comp_mix'] - variables['comp_H_p']
            variables['comp_O_p_err'] = np.sqrt(variables['comp_mix_err']**2 + variables['comp_H_p_err']**2)

            # need to be check when AZ close to 0.
            variables['AZ1'] = variables['AZ1'] % 360.
            variables['AZ2'] = variables['AZ2'] % 360.
            variables['AZ'] = (variables['AZ1'] + variables['AZ2']) / 2
            diff_az = np.abs(variables['AZ1'] - variables['AZ2'])
            variables['AZ'] = np.where(diff_az<180, variables['AZ'], ((variables['AZ1'] + variables['AZ2'] + 360) / 2) % 360)
            variables['EL'] = (variables['EL1'] + variables['EL2']) / 2 

            variables['RANGE'] = np.tile(vars_fh5['range'], [variables['n_e'].shape[0], 1])
            variables['DATETIME'] = dttool.convert_unix_time_to_datetime_cftime(vars_fh5['timestamps'])
            variables['T_e'] = variables['T_i'] * variables['T_r']
            variables['T_e_err'] = variables['T_e'] * np.sqrt((variables['T_i_err'] / variables['T_i']) ** 2
                                                              + (variables['T_r_err'] / variables['T_r']) ** 2)

        self.variables = variables
        self.metadata = metadata

