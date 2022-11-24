# Licensed under the BSD 3-Clause License
# Copyright (C) 2021 GeospaceLab (geospacelab)
# Author: Lei Cai, Space Physics and Astronomy, University of Oulu

__author__ = "Lei Cai"
__copyright__ = "Copyright 2021, GeospaceLab"
__license__ = "BSD-3-Clause License"
__email__ = "lei.cai@oulu.fi"
__docformat__ = "reStructureText"


import netCDF4
import numpy as np
import cftime
import datetime
import re


class Loader:
    def __init__(self, file_path, file_type='ascii', load_data=True):
        self.file_path = file_path
        self.file_type = file_type
        self.variables = {}
        self.metadata = {}
        self.done = False
        if load_data:
            self.load()

    def load(self):

        with open(self.file_path, 'r') as f:
            text = f.read()
            results = re.findall(
                r'^(\d+ \d+ \d+ \d+ \d+ \d+)\s*([+\-\d.]+)\s*([+\-\d.]+)\s*([+\-\d.]+)\s*([+\-\d.]+)\s*([+\-\d.]+)\s*([+\-\d.]+)\s*([+\-\d.]+)',
                text,
                re.M
            )
            results = list(zip(*results))
            # time_array = np.array([(datetime.datetime.strptime(dtstr+'000', "%Y-%m-%d %H:%M:%S.%f")
            #                       - datetime.datetime(1970, 1, 1)) / datetime.timedelta(seconds=1)
            #                       for dtstr in results[0]])

            dts = np.array([datetime.datetime.strptime(dtstr, "%Y %m %d %H %M %S") for dtstr in results[0]])[:, np.newaxis]
            IL = np.array(results[1]).astype(np.float32)[:, np.newaxis]
            IU = np.array(results[2]).astype(np.float32)[:, np.newaxis]
            IE = np.array(results[3]).astype(np.float32)[:, np.newaxis]
            glat_il = np.array(results[4]).astype(np.float32)[:, np.newaxis]
            glon_il = np.array(results[5]).astype(np.float32)[:, np.newaxis]
            glat_iu = np.array(results[6]).astype(np.float32)[:, np.newaxis]
            glon_iu = np.array(results[7]).astype(np.float32)[:, np.newaxis]
            self.variables['DATETIME'] = dts
            self.variables['IL'] = IL
            self.variables['IU'] = IU
            self.variables['IE'] = IE
            self.variables['GEO_LAT_IL'] = glat_il
            self.variables['GEO_LON_IL'] = glon_il
            self.variables['GEO_LAT_IU'] = glat_iu
            self.variables['GEO_LON_IU'] = glon_iu

        with open(self.file_path, 'r') as f:
            text = f.read()
            results = re.findall(
                r'(^%[\s\w]+)',
                text,
                re.M
            )
            self.metadata['headers'] = results


