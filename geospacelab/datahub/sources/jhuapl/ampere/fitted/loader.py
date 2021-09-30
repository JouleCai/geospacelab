# Licensed under the BSD 3-Clause License
# Copyright (C) 2021 GeospaceLab (geospacelab)
# Author: Lei Cai, Space Physics and Astronomy, University of Oulu

import netCDF4
import datetime
import numpy as np
import geospacelab.toolbox.utilities.pydatetime as dttool


class Loader(object):

    def __init__(self, file_path, file_type='fitted', pole='N'):

        self.variables = {}
        self.metadata = {}
        self.file_path = file_path
        self.file_type = file_type
        self.pole = pole

        self.load_data()

    def load_data(self):
        if self.file_type == 'fitted':
            self.load_data_fitted()

    def load_data_fitted(self):
        dataset = netCDF4.Dataset(self.file_path)
        variables = {}

        time_1 = np.array([
            datetime.datetime(yy, mm, dd, HH, MM, SS)
            for yy, mm, dd, HH, MM, SS in zip(
                dataset.variables['start_yr'][::],
                dataset.variables['start_mo'][::],
                dataset.variables['start_dy'][::],
                dataset.variables['start_hr'][::],
                dataset.variables['start_mt'][::],
                dataset.variables['start_sc'][::],
            )
        ])
        time_2 = np.array([
            datetime.datetime(yy, mm, dd, HH, MM, SS)
            for yy, mm, dd, HH, MM, SS in zip(
                dataset.variables['end_yr'][::],
                dataset.variables['end_mo'][::],
                dataset.variables['end_dy'][::],
                dataset.variables['end_hr'][::],
                dataset.variables['end_mt'][::],
                dataset.variables['end_sc'][::],
            )
        ])

        ntime = time_1.shape[0]
        nlon = dataset.variables['nlon'][0]
        nlat = dataset.variables['nlat'][0]
        colat = dataset.variables['colat'][::].reshape((ntime, nlon, nlat))

        mlt = dataset.variables['mlt'][::].reshape((ntime, nlon, nlat))

        Jr = dataset.variables['Jr'][::].reshape(ntime, nlon, nlat)

        variables['DATETIME'] = np.reshape(time_1 + (time_2 - time_1) / 2, (ntime, 1))
        variables['DATETIME_1'] = np.reshape(time_1, (ntime, 1))
        variables['DATETIME_2'] = np.reshape(time_2, (ntime, 1))

        variables['GRID_MLAT'] = np.array(90. - colat)
        variables['GRID_MLT'] = np.array(mlt)

        variables['GRID_Jr'] = np.array(Jr)

        dataset.close()

        self.variables = variables


if __name__ == "__main__":
    import pathlib
    fp = pathlib.Path('/home/lei/afys-data/JHUAPL/AMPERE/Fitted/201603/AMPERE_fitted_20160314.0000.86400.600.north.grd.ncdf')
    loader = Loader(file_path=fp)


    # if hasattr(readObj, 'pole'):
    #    readObj.filter_data_pole(boundinglat = 25)