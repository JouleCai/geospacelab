# Licensed under the BSD 3-Clause License
# Copyright (C) 2021 GeospaceLab (geospacelab)
# Author: Lei Cai, Space Physics and Astronomy, University of Oulu

__author__ = "Lei Cai"
__copyright__ = "Copyright 2021, GeospaceLab"
__license__ = "BSD-3-Clause License"
__email__ = "lei.cai@oulu.fi"
__docformat__ = "reStructureText"

import re

import netCDF4 as nc
import numpy as np
import cftime
import datetime

import geospacelab.toolbox.utilities.pydatetime as dttool


class Loader:
    """
    :param file_path: the file's full path
    :type file_path: pathlib.Path object
    :param file_type: the specific file type for the file being loaded. Options: ['TEC-MAT'], 'TEC-LOS', 'TEC-sites')
    :type file_type: str
    :param load_data: True, load without calling the method "load_data" separately.
    :type load_data: bool
    """
    def __init__(self, file_path, file_type='ATEC', load_data=True):
        self.file_path: pathlib.Path = file_path
        self.file_type = file_type
        self.variables = {}
        self.done = False
        if load_data:
            self.load()

    def load(self):
        if self.file_type == "ATEC":
            self.load_tec_map()
        else:
            raise NotImplemented

    def load_tec_map(self):
        fnc = nc.Dataset(self.file_path)
        file_name = self.file_path.stem
        rc = re.compile(r"(\d{10})")
        tstr = rc.search(file_name).groups()[0]
        dt_0 = dttool.get_start_of_the_day(datetime.datetime.strptime(tstr, "%Y%m%d%H"))

        variables = {}

        glat = np.array(fnc.variables['lat'])
        glon = np.array(fnc.variables['lon'])
        variables['GEO_LAT'] = np.tile(glat.reshape(glat.size, 1), (1, glon.size))
        variables['GEO_LON'] = np.tile(glon.reshape(1, glon.size), (glat.size, 1))
        tec_map = np.array(fnc.variables['atec']).transpose((2, 0, 1))
        tec_map = np.where(tec_map >= 900., np.nan, tec_map)
        variables['TEC_MAP'] = tec_map
        time = np.array(fnc.variables['time'])
        dts = np.array(
            [dt_0 + datetime.timedelta(seconds=t) for t in time]
        )
        variables['DATETIME'] = dts[:, np.newaxis]

        self.variables = variables
        self.done = True
        fnc.close()


nc_variable_name_dict = {
    'GEO_LAT': 'lat',
    'GEO_LON': 'lon',
    'TEC_MAP': 'atec',
}


if __name__ == "__main__":
    import pathlib
    fp = pathlib.Path("/home/lei/Downloads/2023010702_atec.nc")
    Loader(file_path=fp)
    # import pathlib
    # import geospacelab.datahub.sources.madrigal as madrigal
    # file_path = pathlib.Path("/Users/lcai/Downloads/gps200102g.002.hdf5")
    # madrigal.utilities.show_hdf5_structure(filename=file_path.name, filepath= file_path.parent.resolve())
    # madrigal.utilities.show_hdf5_metadata(filename=file_path.name, filepath=file_path.parent.resolve())
    # with h5py.File(file_path, 'r') as fh5:
    #
    #     print(fh5['Data']['Array Layout']['2D Parameters']['tec'][:])  # shape (180, 360, 288)
    #     print(fh5['Data']['Array Layout']['timestamps'][:])  # shape (288) unix time
    #     print(fh5['Data']['Array Layout']['glon'][:])  # gdlat -90. - 89. glon -180. 79.