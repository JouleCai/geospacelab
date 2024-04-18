# Licensed under the BSD 3-Clause License
# Copyright (C) 2021 GeospaceLab (geospacelab)
# Author: Lei Cai, Space Physics and Astronomy, University of Oulu

import netCDF4
import datetime
import numpy as np
import re
import cdflib

import geospacelab.toolbox.utilities.pydatetime as dttool
import geospacelab.toolbox.utilities.pybasic as pybasic


class Loader(object):

    def __init__(self, file_path, file_type='sdr-disk', pole='N', pp_type='AURORAL'):

        self.variables = {}
        self.metadata = {}
        self.file_path = file_path
        self.file_type = file_type
        self.pole = pole
        self.pp_type = pp_type

        self.load_data()

    def load_data(self):
        dataset = netCDF4.Dataset(self.file_path)
        variables = {}
        metadata = {}

        # get the version of the file
        variables['FILE_VERSION'] = re.findall(r"(V[A-Z\d]+S[A-Z\d]+C[A-Z0-9]+)", dataset.FILENAME)[0]
        variables['DATA_PRODUCT_VERSION'] = str(dataset.DATA_PRODUCT_VERSION)
        variables['SOFTWARE_VERSION_NUMBER'] = str(dataset.SOFTWARE_VERSION_NUMBER)
        variables['CALIBRATION_PERIOD_VERSION'] = dataset.CALIBRATION_PERIOD_VERSION
        variables['EMISSION_SPECTRA'] = ['1216', '1304', '1356', 'LBHS', 'LBHL']

        if self.pole == 'N':
            pole = self.pole
            pole_str = 'NORTH'
        elif self.pole == 'S':
            pole = self.pole
            pole_str = 'SOUTH'
        else:
            raise ValueError

        if self.pp_type in ['AURORAL', 'DAY_AURORAL', 'DAY_AUR']:
            pp_key_1 = 'DAY'
            pp_key_2 = 'AURORAL'
        else:
            pp_key_1 = self.pp_type
            pp_key_2 = ''

        # Global attributes
        starting_time = datetime.datetime.strptime(dataset.STARTING_TIME, "%Y%j%H%M%S")
        variables['STARTING_TIME'] = starting_time
        stopping_time = datetime.datetime.strptime(dataset.STOPPING_TIME, "%Y%j%H%M%S")
        variables['STOPPING_TIME'] = stopping_time
        dt0 = dttool.get_start_of_the_day(starting_time)

        # Time and position
        sectime = pybasic.str_join('TIME', pp_key_1, pp_key_2, separator='_', uppercase=True)
        sectime = np.array(dataset.variables[sectime])
        cdfepoch = pybasic.str_join('TIME', 'EPOCH', pp_key_1, pp_key_2, separator='_', uppercase=True)
        dts = cdflib.cdfepoch.unixtime(np.array(dataset.variables[cdfepoch]))
        dts = dttool.convert_unix_time_to_datetime_cftime(dts)

        var_name_nc = pybasic.str_join('ORBIT', pp_key_1, pp_key_2, separator='_', uppercase=True)
        variables['SC_ORBIT_ID'] = np.array(dataset.variables[var_name_nc])

        var_name_nc = pybasic.str_join('LATITUDE', pp_key_1, pp_key_2, separator='_', uppercase=True)
        geo_lat = np.array(dataset.variables[var_name_nc])

        ind_full = np.arange(0, geo_lat.shape[0])
        if self.pole in ['N', 'S']:
            ind_S = np.where(geo_lat < -5.)[0]
            if list(ind_S):
                ind_N = ind_full[0:ind_S[0]]
            else:
                ind_N = np.where(geo_lat > 5.)[0]
            if (self.pole == 'N' and not list(ind_N)) or (self.pole == 'S' and not list(ind_S)):
                raise ValueError("No data available!")

            if self.pole == 'N':
                ind_along = ind_N
            else:
                ind_along = ind_S
        elif self.pole == 'both':
            ind_along = ind_full
        else:
            raise NotImplementedError

        variables['SC_DATETIME'] = dts[ind_along]

        variables['SC_GEO_LAT'] = geo_lat[ind_along]

        if self.pole in ['N', 'S']:
            ind_center = np.where(np.abs(variables['SC_GEO_LAT']) == np.max(np.abs(variables['SC_GEO_LAT'])))[0][0]
            variables['DATETIME'] = variables['SC_DATETIME'][ind_center]
        else:
            variables['DATETIME'] = starting_time + (stopping_time - starting_time) / 2

        var_name_nc = pybasic.str_join('LONGITUDE', pp_key_1, pp_key_2, separator='_', uppercase=True)
        variables['SC_GEO_LON'] = np.array(dataset.variables[var_name_nc])[ind_along]

        var_name_nc = pybasic.str_join('ALTITUDE', pp_key_1, pp_key_2, separator='_', uppercase=True)
        variables['SC_GEO_ALT'] = np.array(dataset.variables[var_name_nc])[ind_along]

        var_name_nc = pybasic.str_join('PIERCEPOINT', pp_key_1, 'LATITUDE', pp_key_2, separator='_', uppercase=True)
        variables['DISK_GEO_LAT'] = np.array(dataset.variables[var_name_nc]).T[ind_along, :]

        var_name_nc = pybasic.str_join('PIERCEPOINT', pp_key_1, 'LONGITUDE', pp_key_2, separator='_', uppercase=True)
        variables['DISK_GEO_LON'] = np.array(dataset.variables[var_name_nc]).T[ind_along, :]

        var_name_nc = pybasic.str_join('PIERCEPOINT', pp_key_1, 'ALTITUDE', pp_key_2, separator='_', uppercase=True)
        variables['DISK_GEO_ALT'] = np.array(dataset.variables[var_name_nc])

        var_name_nc = pybasic.str_join('PIERCEPOINT', pp_key_1, 'SZA', pp_key_2, separator='_', uppercase=True)
        variables['DISK_SZA'] = np.array(dataset.variables[var_name_nc]).T[ind_along, :]

        var_name_nc = pybasic.str_join('IN_SAA', pp_key_1, pp_key_2, separator='_', uppercase=True)
        variables['DISK_SAA'] = np.array(dataset.variables[var_name_nc]).T[ind_along, :]

        var_name_nc = pybasic.str_join('ACROSSPIXELSIZE', pp_key_1, pp_key_2, separator='_', uppercase=True)
        variables['ACROSS_PIXEL_SIZE'] = np.array(dataset.variables[var_name_nc])

        var_name_nc = pybasic.str_join('ALONGPIXELSIZE', pp_key_1, pp_key_2, separator='_', uppercase=True)
        variables['ALONG_PIXEL_SIZE'] = np.array(dataset.variables[var_name_nc])

        var_name_nc = pybasic.str_join('EFFECTIVELOOKANGLE', pp_key_1, pp_key_2, separator='_', uppercase=True)
        variables['EFFECTIVE_LOOK_ANGLE'] = np.array(dataset.variables[var_name_nc]).T[ind_along, :]

        # Re-binned data
        # var_name_nc = pybasic.str_join('DISKCOUNTSDATA', pp_key_1, pp_key_2, separator='_', uppercase=True)
        # variables['DISK_COUNTS_DATA'] = np.array(dataset.variables[var_name_nc])

        # var_name_nc = pybasic.str_join('DISKDECOMP_UNCERTAINTY', pp_key_1, pp_key_2, separator='_', uppercase=True)
        # variables['DISK_DECOMP_UNCERTAINTY'] = np.array(dataset.variables[var_name_nc])

        var_name_nc = pybasic.str_join('EXPOSURE', pp_key_1, pp_key_2, separator='_', uppercase=True)
        variables['EXPOSURE'] = np.array(dataset.variables[var_name_nc]).T[ind_along, :]

        var_name_nc = pybasic.str_join('SAA_COUNT', pp_key_1, pp_key_2, separator='_', uppercase=True)
        variables['SAA_COUNT'] = np.array(dataset.variables[var_name_nc]).T[ind_along, :]

        # Calibration parameters
        var_name_nc = 'DARK_COUNT_CORRECTION'
        variables['DARK_COUNT_CORRECTION'] = np.array(dataset.variables[var_name_nc])

        var_name_nc = 'SCATTER_LIGHT_1216_CORRECTION'
        variables['SCATTER_LIGHT_1216_CORRECTION'] = np.array(dataset.variables[var_name_nc])

        var_name_nc = 'SCATTER_LIGHT_1304_CORRECTION'
        variables['SCATTER_LIGHT_1304_CORRECTION'] = np.array(dataset.variables[var_name_nc])

        var_name_nc = 'OVERLAP_1304_1356_CORRECTION'
        variables['OVERLAP_1304_1356_CORRECTION'] = np.array(dataset.variables[var_name_nc])

        var_name_nc = 'LONGWAVE_SCATTER_CORRECTION'
        variables['LONG_WAVE_SCATTER_CORRECTION'] = np.array(dataset.variables[var_name_nc])

        var_name_nc = 'RED_LEAK_CORRECTION'
        variables['RED_LEAK_CORRECTION'] = np.array(dataset.variables[var_name_nc])

        # Calibrated data
        var_name_nc = pybasic.str_join('DQI', pp_key_1, pp_key_2, separator='_', uppercase=True)
        variables['DQI'] = np.array(dataset.variables[var_name_nc]).T[ind_along, :]

        dict_vn_nc = {
            'DISK_COUNTS':  pybasic.str_join('DISKCOUNTSDATA', pp_key_1, pp_key_2, separator='_', uppercase=True),
            'DISK_R': pybasic.str_join('DISK', 'INTENSITY', pp_key_1, pp_key_2, separator='_', uppercase=True),
            'DISK_R_RECT': pybasic.str_join('DISK', 'RECTIFIED', 'INTENSITY', pp_key_1, pp_key_2, separator='_', uppercase=True),
            'DISK_DECOMP_ERROR': pybasic.str_join('DISKDECOMP_UNCERTAINTY', pp_key_1, pp_key_2, separator='_', uppercase=True),
            'DISK_R_ERROR': pybasic.str_join('DISK', 'RADIANCE', 'UNCERTAINTY', pp_key_1, pp_key_2,
                                                  separator='_', uppercase=True),
            'DISK_R_RECT_ERROR': pybasic.str_join('DISK', 'RECTIFIED', 'RADIANCE', 'UNCERTAINTY', pp_key_1, pp_key_2,
                                                  separator='_', uppercase=True),
            'DISK_CALIB_ERROR': pybasic.str_join('DISK', 'CALIBRATION', 'UNCERTAINTY', pp_key_1, pp_key_2,
                                                  separator='_', uppercase=True),
            'DQI': pybasic.str_join('DQI', pp_key_1, pp_key_2, 'CHAN', separator='_', uppercase=True),
        }

        for vnp, vn_nc in dict_vn_nc.items():
            disk_array = np.array(dataset.variables[vn_nc])
            for ind, emline in enumerate(variables['EMISSION_SPECTRA']):
                if 'ERROR' in vnp:
                    vn = vnp.replace('_ERROR', '') +  '_' + emline + '_ERROR'
                else:
                    vn = vnp + '_' + emline

                variables[vn] = disk_array[:, :, ind].T[ind_along, :]

        dataset.close()

        self.variables = variables
        self.metadata = metadata
