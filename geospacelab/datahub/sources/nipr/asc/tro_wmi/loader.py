# Licensed under the BSD 3-Clause License
# Copyright (C) 2021 GeospaceLab (geospacelab)
# Author: Lei Cai, Space Physics and Astronomy, University of Oulu
import re

import netCDF4
import datetime
import numpy as np
import matplotlib.pyplot as plt
import geospacelab.toolbox.utilities.pydatetime as dttool


class Loader(object):

    def __init__(self, file_path, file_ext='jpg', channel='558'):

        self.variables = {}
        self.metadata = {}
        self.file_path = file_path
        self.file_ext = file_ext
        self.channel = channel

        self.load_data()

    def load_data(self):
        img = plt.imread(self.file_path)

        if self.channel=='color':
            self.variables['ASC_IMG_DATA'] = img
        else:
            self.variables['ASC_IMG_DATA'] = img[:, :, 1]

        # extract datetime from the file name
        file_name = self.file_path.name
        rc = re.search('\d{2}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}-\d{2}', file_name)
        self.variables['DATETIME'] = datetime.datetime.strptime(rc.group()+'0000', "%y-%m-%d_%H-%M-%S-%f")

        file_dir = self.file_path.parent.parent.resolve()
        try:
            file_path = list(file_dir.glob('*el.txt'))[0]
            self.variables['ASC_IMG_EL'] = np.loadtxt(file_path)
            file_path = list(file_dir.glob('*az.txt'))[0]
            self.variables['ASC_IMG_AZ'] = np.loadtxt(file_path)
        except Exception as a:
            print('The files recording the azimuth/elevation angles are not found!')
            self.variables['ASC_IMG_EL'] = None
            self.variables['ASC_IMG_AZ'] = None

        metadata = {}
        self.metadata = metadata


if __name__ == "__main__":
    import pathlib
    obj = Loader(file_path=pathlib.Path('/home/lei/afys-data/NIPR/ASC/TRO/2018/20181107/558nm/20181107_20/wat_tr2_18-11-07_20-00-27-62.jpg'))