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


var_name_dict = {
    'AZ': 'azm',
    'EL': 'elm',
    'PULSE_LENGTH': 'pl',
    'P_Tx': 'power',
    'n_e': 'ne',
    'n_e_err': 'dne',
    'T_i': 'ti',
    'T_i_err': 'dti',
    'T_e': 'te',
    'T_e_err': 'dte',
    'v_i_los': 'vo',
    'v_i_los_err': 'dvo',
    'comp_O_p': 'po+',
    'comp_O_p_err': 'dpo+',
    'CGM_LAT': 'cgm_lat',
    'CGM_LON': 'cgm_long',
    'HEIGHT': 'gdalt',
    'BEAM_ID': 'beamid',
    'CHISQ': 'chisq',
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
    def __init__(self, file_path, beam_id=None, beam_az=None, beam_el=None, direct_load=True):
        self.file_path = file_path
        self.beam_id = beam_id
        self.beam_az = beam_az
        self.beam_el = beam_el
        self.beams = None
        self.variables = {}
        self.metadata = {}

        self.done = False
        if direct_load:
            self.load()
    
    def list_beams(self, display=True):
        with h5py.File(self.file_path, 'r') as fh5:
            data_fh5 = fh5['Data']
            array_layouts = list(data_fh5['Array Layout'].keys())
            
            beam_ids = []
            beam_azs = []
            beam_els = []
            for i, array_layout_str in enumerate(array_layouts):
                
                res = re.search(r'beamid=([\d.]+)', array_layout_str)
                beam_id = int(float(res.groups()[0]))
                beam_az = np.nanmedian(data_fh5['Array Layout'][array_layout_str]['1D Parameters']['azm'])
                beam_el = np.nanmedian(data_fh5['Array Layout'][array_layout_str]['1D Parameters']['elm'])

                beam_ids.append(beam_id)
                beam_azs.append(beam_az)
                beam_els.append(beam_el)
        beam_azs = [beam_az % 360 for beam_az in beam_azs]
        beam_data = {
            'IDs': beam_ids,
            'AZs': beam_azs,
            'ELs': beam_els,
        }
        self.beams = beam_data
        
        if display:
            mylog.simpleinfo.info("Listing the beams ...")
            mylog.simpleinfo.info("{:>10s}{:>12s}{:>12.2s}{:>12.2s}".format('Num', 'ID', 'AZ', 'EL'))
            for i, beam_id in enumerate(beam_ids):
                mylog.simpleinfo.info(
                    "{:>10d}{:>12d}{:>12.2f}{:>12.2f}".format(
                        i + 1,
                        beam_id,
                        beam_azs[i],
                        beam_els[i]
                    )
                )
        return 
                
    def select_beam(self, az=None, el=None, error_az=0.2, error_el=0.2):
        
        beam_azs = np.array(self.beams['AZs'])
        beam_els = np.array(self.beams['ELs'])
        beam_ids = np.array(self.beams['IDs'])
        
        ind_1 = np.where(((np.abs(beam_azs - az) <= error_az) & (np.abs(beam_els - el) <= error_el)))[0]
        ind_2 = np.where(((np.abs(beam_azs - 360. - az) <= error_az) & (np.abs(beam_els - el) <= error_el)))[0] 
        ind_1 = np.append(ind_1, ind_2) 
        
        if not list(ind_1):
            mylog.StreamLogger.error(f"Cannot find the beam with AZ={az} and EL={el}!")
            raise AttributeError
        
        if len(ind_1) > 2:
            mylog.StreamLogger.error(f"Multiple beams found to fulfill AZ={az} and EL={el}!")
            raise AttributeError
        
        self.beam_id = beam_ids[ind_1[0]]
        self.beam_az = beam_azs[ind_1[0]]
        self.beam_el = beam_els[ind_1[0]]
        mylog.simpleinfo.info(f'Select the beam: ID={self.beam_id}, AZ={self.beam_az}, EL={self.beam_el}')
        return self.beam_id
    
    def get_metadata(self):
        with h5py.File(self.file_path, 'r') as fh5:
            metadata_fh5 = fh5['Metadata'] 
                
            exp_data = metadata_fh5['Experiment Parameters'][::]
            metadata = {}
            for (key, value) in exp_data:
                metadata[key.decode('UTF-8').strip()] = value.decode('UTF-8').strip()
             
            metadata['NOTES'] = [item[0].decode('UTF-8') for item in metadata_fh5['Experiment Notes'][::]]
        
        return metadata

    def load_from_table_layout(self):
        variables = {}
        metadata = {}



    def load(self):
        variables = {}
        metadata = {}
        
        if self.beam_id is None:
            self.list_beams(display=True)
            
            self.select_beam(az=self.beam_az, el=self.beam_el)
        
        with h5py.File(self.file_path, 'r') as fh5:
            data_fh5 = fh5['Data']
            array_layouts = list(data_fh5['Array Layout'].keys())

            matching = 0
            for array_layout_str in array_layouts:
                if str(self.beam_id) in array_layout_str:
                    matching = 1
                    break
                
            if matching == 0:
                raise AttributeError

            vars_fh5 = {}
            fh5_vars_1d = data_fh5['Array Layout'][array_layout_str]['1D Parameters']
            for var_name in fh5_vars_1d.keys():
                if var_name == 'Data Parameters':
                    continue
                vars_fh5[var_name] = np.array(fh5_vars_1d[var_name])[:, np.newaxis]
            fh5_vars_2d = data_fh5['Array Layout'][array_layout_str]['2D Parameters']
            for var_name in fh5_vars_2d.keys():
                if var_name == 'Data Parameters':
                    continue
                if var_name == 'nel':
                    vars_fh5['ne'] = 10**np.array(fh5_vars_2d[var_name]).T
                elif var_name == 'dnel':
                    vars_fh5['dne'] = 10 ** np.array(fh5_vars_2d[var_name]).T
                else:
                    vars_fh5[var_name] = np.array(fh5_vars_2d[var_name]).T
            vars_fh5['range'] = np.array(data_fh5['Array Layout'][array_layout_str]['range'])[np.newaxis, :]
            if np.median(vars_fh5['range'].flatten()) > 1e5:
                mylog.StreamLogger.warning(f"The variable range is detected in [m]. It is converted into [km].")
                vars_fh5['range'] = vars_fh5['range'] * 1e-3
            vars_fh5['timestamps'] = np.array(data_fh5['Array Layout'][array_layout_str]['timestamps'])[:, np.newaxis]
            for var_name, var_name_fh5 in var_name_dict.items():
                if var_name_fh5 not in vars_fh5.keys():
                    mylog.StreamLogger.warning(f"The requested variable {var_name_fh5} does not exist in the data file!")
                    variables[var_name] = None
                    continue
                variables[var_name] = vars_fh5[var_name_fh5]

            variables['comp_mix'] = 1. - variables['comp_O_p']
            variables['comp_mix_err'] = variables['comp_O_p_err']

            variables['RANGE'] = np.tile(vars_fh5['range'], [variables['n_e'].shape[0], 1])
            variables['DATETIME'] = dttool.convert_unix_time_to_datetime_cftime(vars_fh5['timestamps'])
                
            self.variables = variables
            
            metadata = self.get_metadata()
            
            if 'alternating' in metadata['kind of data file'].lower():
                pulse_code = 'Alternating Code'
            elif 'long pulse' in metadata['kind of data file'].lower():
                pulse_code = 'Long Pulse'
            else:
                pulse_code = ''
            
            pulse_length = np.nanmedian(variables['PULSE_LENGTH'].flatten())
            
            metadata['PULSE_CODE'] = pulse_code
            metadata['PULSE_LENGTH'] = pulse_length

            self.beam_az = np.nanmedian(variables['AZ'])
            self.beam_el = np.nanmedian(variables['EL'])
            
            self.metadata = metadata
