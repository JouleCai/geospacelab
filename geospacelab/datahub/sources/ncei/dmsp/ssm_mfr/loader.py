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
import re

import numpy as np

import geospacelab.toolbox.utilities.pylogging as mylog
import geospacelab.toolbox.utilities.pydatetime as dttool


class Loader(object):
    def __init__(self, file_path, direct_load=True, **kwargs):

        self.file_path = pathlib.Path(file_path)
        self.variables = {}
        self.metadata = {}

        if direct_load:
            self.load_data()

    def load_data(self):

        with open(self.file_path, 'r') as f:
            text = f.read()
            results = re.findall(
                r'^(\d{4})(\d{3})(\d{2})(\d{2})([\d.]+)\s(.{10})' +
                r'\s*(\d+)\s*(\w)\s*([\-\d.]+)\s*([\-\d.]+)\s*([\d.]+)\s*(\w)\s*(\w)' +
                r'\s*([\-\d]+)\s*([\-\d]+)\s*([\-\d]+)\s*(\d+)\.(\d+)' +
                r'\s*([\-\d]+)\s*([\-\d]+)\s*([\-\d]+)\s*(\d+)\.(\d+)\s*(\w)\n',
                text,
                re.M
            )
            results = list(zip(*results))

        seconds = [int(np.floor(float(s))) for s in results[4]]
        microseconds = [int((float(s) - np.floor(float(s)))*1e6) for s in results[4]]
        doys = np.array([int(d) for d in results[1]])
        year = int(results[0][0])
        dates = dttool.convert_doy_to_datetime(year, doys)
        hours = [int(np.floor(float(h))) for h in results[2]]
        minutes = [int(np.floor(float(m))) for m in results[3]] 
        dts = np.array([
            d + datetime.timedelta(hours=hh, minutes=mm, seconds=ss, microseconds=ms)
            for d, hh, mm, ss, ms in zip(dates, hours, minutes, seconds, microseconds)
        ])

        num_rec = len(dts)
        self.variables['SC_DATETIME'] = np.array(dts).reshape(num_rec, 1)
        self.variables['SC_GEO_ALT'] = np.array(results[10]).astype(np.float32).reshape(num_rec, 1)  # in km
        self.variables['SC_GEO_LON'] = np.array(results[9]).astype(np.float32).reshape(num_rec, 1)
        self.variables['SC_GEO_LAT'] = np.array(results[8]).astype(np.float32).reshape(num_rec, 1)
        self.variables['B_D'] = np.array(results[13]).astype(np.float32).reshape(num_rec, 1) * 1e-9
        self.variables['B_F'] = np.array(results[14]).astype(np.float32).reshape(num_rec, 1) * 1e-9 
        self.variables['B_P'] = np.array(results[15]).astype(np.float32).reshape(num_rec, 1) * 1e-9
        self.variables['d_B_D'] = np.array(results[18]).astype(np.float32).reshape(num_rec, 1) * 1e-9
        self.variables['d_B_F'] = np.array(results[19]).astype(np.float32).reshape(num_rec, 1) * 1e-9
        self.variables['d_B_P'] = np.array(results[20]).astype(np.float32).reshape(num_rec, 1) * 1e-9
        return


if __name__ == "__main__":
    load_obj = Loader(
        file_path="/home/lei/afys-data/NCEI/DMSP/SSM_MFR/F18/2024/05/PS.CKGWC_SC.U_DI.A_GP.SSMXX-F18-R99990-B9999090-APSM_AR.GLOBAL_DD.20240510_TP.000001-235958_DF.MFR"
    )