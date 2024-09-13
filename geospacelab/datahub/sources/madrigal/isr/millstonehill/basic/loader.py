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
            self.load()

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
            variables['AZ'] = variables['AZ1'] % 360.
            variables['EL'] = variables['EL1']

            variables['RANGE'] = np.tile(vars_fh5['range'], [variables['n_e'].shape[0], 1])
            variables['DATETIME'] = dttool.convert_unix_time_to_datetime_cftime(vars_fh5['timestamps'])
            variables['T_e'] = variables['T_i'] * variables['T_r']
            variables['T_e_err'] = variables['T_e'] * np.sqrt((variables['T_i_err'] / variables['T_i']) ** 2
                                                              + (variables['T_r_err'] / variables['T_r']) ** 2)

        self.variables = variables
        self.metadata = metadata


if __name__ == "__main__":
    import pathlib
    fp = pathlib.Path("/home/lei/afys-data/Madrigal/Millstone_ISR/2016/20160314/Millstone_ISR_combined_20160314.005.hdf5")
    Loader(file_path=fp)