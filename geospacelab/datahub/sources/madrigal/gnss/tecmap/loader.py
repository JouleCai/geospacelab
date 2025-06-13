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
import datetime


class Loader:
    """
    :param file_path: the file's full path
    :type file_path: pathlib.Path object
    :param file_type: the specific file type for the file being loaded. Options: ['TEC-MAT'], 'TEC-LOS', 'TEC-sites')
    :type file_type: str
    :param load_data: True, load without calling the method "load_data" separately.
    :type load_data: bool
    """
    def __init__(self, file_path, file_type='TEC-MAP', load_data=True):
        self.file_path = file_path
        self.file_type = file_type
        self.variables = {}
        self.done = False
        if load_data:
            self.load()

    def load(self):
        if self.file_type == "TEC-MAP":
            self.load_tec_map()
        else:
            raise NotImplemented

    def load_tec_map(self):
        variables = {}
        with h5py.File(self.file_path, 'r') as fh5:
            tec = fh5['Data']['Array Layout']['2D Parameters']['tec'][:]
            variables['TEC_MAP'] = np.transpose(tec, (2, 0, 1))

            t = fh5['Data']['Array Layout']['timestamps'][:]
            t = cftime.num2date(
                t,
                units="seconds since 1970-01-01 00:00:00.0",
                only_use_cftime_datetimes=False,
                only_use_python_datetimes=True,
            )
            variables['DATETIME'] = np.reshape(t, (t.size, 1))

            glon =fh5['Data']['Array Layout']['glon'][:]
            glat = fh5['Data']['Array Layout']['gdlat'][:]
            variables['GEO_LAT'] = np.tile(glat.reshape(glat.size, 1), (1, glon.size))
            variables['GEO_LON'] = np.tile(glon.reshape(1, glon.size), (glat.size, 1))
        self.variables = variables



if __name__ == "__main__":
    import pathlib
    fp = pathlib.Path("/Users/lcai/Downloads/gps200102g.002.hdf5")
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