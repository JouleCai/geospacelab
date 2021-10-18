# Licensed under the BSD 3-Clause License
# Copyright (C) 2021 GeospaceLab (geospacelab)
# Author: Lei Cai, Space Physics and Astronomy, University of Oulu

import netCDF4
import datetime
import numpy as np
import re
import geospacelab.toolbox.utilities.pydatetime as dttool



class Loader(object):

    def __init__(self, file_path, file_type='nc', pole='N'):

        self.variables = {}
        self.metadata = {}
        self.file_path = file_path
        self.file_type = file_type
        self.pole = pole

        self.load_data()

    def load_data(self):
        if self.file_type == 'nc':
            self.load_data_nc()

    def load_data_nc(self):

        dataset = netCDF4.Dataset(self.file_path)
        variables = {}

        unix_time = dataset.variables['UNIX_TIME'][::]
        dts = dttool.convert_unix_time_to_datetime_cftime(unix_time)
        ntime = dts.size
        variables['DATETIME'] = np.reshape(dts, (ntime, 1))

        variables['GRID_MLAT'] = np.array(dataset.variables['MLAT'][::])

        dataset.close()

        self.variables = variables


if __name__ == "__main__":
    import pathlib
    fp = pathlib.Path('/home/lei/afys-data/SuperDARN/PotentialMap/2016/SuperDARN_POTMAP_2min_20160315_N.nc')
    loader = Loader(file_path=fp)


    # if hasattr(readObj, 'pole'):
    #    readObj.filter_data_pole(boundinglat = 25)