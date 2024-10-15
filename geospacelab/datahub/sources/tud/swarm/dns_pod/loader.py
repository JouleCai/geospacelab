# Licensed under the BSD 3-Clause License
# Copyright (C) 2021 GeospaceLab (geospacelab)
# Author: Lei Cai, Space Physics and Astronomy, University of Oulu

__author__ = "Lei Cai"
__copyright__ = "Copyright 2021, GeospaceLab"
__license__ = "BSD-3-Clause License"
__email__ = "lei.cai@oulu.fi"
__docformat__ = "reStructureText"

import re
import numpy as np
import datetime


class Loader(object):

    def __init__(self, file_path, file_type='txt', version='v01', direct_load=True):
        self.file_path = file_path
        self.file_type = file_type
        self.version = version
        self.variables = {}
        self.done = False
        if direct_load:
            self.load()

    def load(self):

        if self.version == 'v01':
            self.load_v01()
        else:
            raise NotImplementedError

    def load_v01(self):
        with open(self.file_path, 'r') as f:
            text = f.read()
            results = re.findall(
            r"^(\d{4}-\d{2}-\d{2}\s*\d{2}\:\d{2}\:\d{2}\.\d{3})\s*\w{3}\s*([\-\d.]+)\s*([\-\d.]+)\s*([\-\d.]+)\s*([\-\d.]+)\s*([\-\d.]+)\s*([+\-\d.Ee]{3,})\s*",
            text,
            re.M)
            results = list(zip(*results))
            dts = [datetime.datetime.strptime(dtstr + '000', "%Y-%m-%d %H:%M:%S.%f") for dtstr in results[0]]
            num_rec = len(dts)
            self.variables['SC_DATETIME'] = np.array(dts).reshape(num_rec, 1)
            self.variables['SC_GEO_ALT'] = np.array(results[1]).astype(np.float32).reshape(num_rec, 1) * 1e-3   # in km
            self.variables['SC_GEO_LON'] = np.array(results[2]).astype(np.float32).reshape(num_rec, 1)
            self.variables['SC_GEO_LAT'] = np.array(results[3]).astype(np.float32).reshape(num_rec, 1)
            self.variables['SC_GEO_LST'] = np.array(results[4]).astype(np.float32).reshape(num_rec, 1)
            self.variables['SC_ARG_LAT'] = np.array(results[5]).astype(np.float32).reshape(num_rec, 1)
            self.variables['rho_n'] = np.array(results[6]).astype(np.float32).reshape(num_rec, 1)
