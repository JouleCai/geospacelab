import netCDF4
# from cdf.internal import EPOCHbreakdown
import os
import datetime
import numpy as np


class Loader(object):

    def __init__(self, file_path, file_type='edr-aur', pole = 'N'):

        self.variables = {}
        self.metadata = {}
        self.file_path = file_path
        self.file_type = file_type
        self.pole = pole

        self.load_data()

    def load_data(self):
        if self.file_type == 'sdr':
            self.load_data_sdr()
        elif self.file_type == 'edr-aur':
            self.load_data_edr_aur()
        else:
            raise NotImplemented

    def load_data_sdr(self):
        pass

    def load_data_edr_aur(self):
        dataset = netCDF4.Dataset(self.file_path)
        variables = {}

        # Time and Position
        variables['EFFECTIVE_TIME'] = np.array(dataset.variables['TIME'])
        variables['SC_LAT'] = np.array(dataset.variables['LATITUDE'])
        variables['SC_LON'] = np.array(dataset.variables['LONGITUDE'])
        variables['SC_ALT'] = np.array(dataset.variables['ALTITUDE'])

        # Auroral map, #colors: 0: '1216', 1: '1304', 2: '1356', 3: 'LBHS', 4: 'LBHL'.
        variables['GRID_S_AUR'] = np.array(dataset.variables['DISK_RADIANCEDATA_INTENSITY_SOUTH']).transpose((0, 2, 1))
        variables['GRID_N_AUR'] = np.array(dataset.variables['DISK_RADIANCEDATA_INTENSITY_NORTH']).transpose((0, 2, 1))
        variables['GRID_S_MLAT'] = np.array(dataset.variables['LATITUDE_GEOMAGNETIC_GRID_MAP']).T
        variables['GRID_N_MLAT'] = np.array(dataset.variables['LATITUDE_GEOMAGNETIC_GRID_MAP']).T
        variables['GRID_S_MLON'] = np.array(dataset.variables['LONGITUDE_GEOMAGNETIC_SOUTH_GRID_MAP']).T
        variables['GRID_N_MLON'] = np.array(dataset.variables['LONGITUDE_GEOMAGNETIC_NORTH_GRID_MAP']).T
        variables['GRID_S_UT'] = np.array(dataset.variables['UT_S']).transpose((0, 2, 1)).T
        variables['GRID_N_UT'] = np.array(dataset.variables['UT_N']).transpose((0, 2, 1)).T

        # Auroral oval boundary
        variables['AOB_N_EQ_MLAT'] = np.array(dataset.variables['NORTH_GEOMAGNETIC_LATITUDE'])
        variables['AOB_N_EQ_MLON'] = np.array(dataset.variables['NORTH_GEOMAGNETIC_LONGITUDE'])
        variables['AOB_N_EQ_MLT'] = np.array(dataset.variables['NORTH_MAGNETIC_LOCAL_TIME'])

        variables['AOB_N_PL_MLAT'] = np.array(dataset.variables['NORTH_POLAR_GEOMAGNETIC_LATITUDE'])
        variables['AOB_N_PL_MLON'] = np.array(dataset.variables['NORTH_POLAR_GEOMAGNETIC_LONGITUDE'])
        variables['AOB_N_PL_MLT'] = np.array(dataset.variables['NORTH_POLAR_MAGNETIC_LOCAL_TIME'])

        variables['AOB_S_EQ_MLAT'] = np.array(dataset.variables['SOUTH_GEOMAGNETIC_LATITUDE'])
        variables['AOB_S_EQ_MLON'] = np.array(dataset.variables['SOUTH_GEOMAGNETIC_LONGITUDE'])
        variables['AOB_S_EQ_MLT'] = np.array(dataset.variables['SOUTH_MAGNETIC_LOCAL_TIME'])

        variables['AOB_S_PL_MLAT'] = np.array(dataset.variables['SOUTH_POLAR_GEOMAGNETIC_LATITUDE'])
        variables['AOB_S_PL_MLON'] = np.array(dataset.variables['SOUTH_POLAR_GEOMAGNETIC_LONGITUDE'])
        variables['AOB_S_PL_MLT'] = np.array(dataset.variables['SOUTH_POLAR_MAGNETIC_LOCAL_TIME'])

        variables['MAOB_N_EQ_MLAT'] = np.array(dataset.variables['MODEL_NORTH_GEOMAGNETIC_LATITUDE'])
        variables['MAOB_N_EQ_MLON'] = np.array(dataset.variables['MODEL_NORTH_GEOMAGNETIC_LONGITUDE'])
        variables['MAOB_N_EQ_MLT'] = np.array(dataset.variables['MODEL_NORTH_MAGNETIC_LOCAL_TIME'])

        variables['MAOB_N_PL_MLAT'] = np.array(dataset.variables['MODEL_NORTH_POLAR_GEOMAGNETIC_LATITUDE'])
        variables['MAOB_N_PL_MLON'] = np.array(dataset.variables['MODEL_NORTH_POLAR_GEOMAGNETIC_LONGITUDE'])
        variables['MAOB_N_PL_MLT'] = np.array(dataset.variables['MODEL_NORTH_POLAR_MAGNETIC_LOCAL_TIME'])

        variables['MAOB_S_EQ_MLAT'] = np.array(dataset.variables['MODEL_SOUTH_GEOMAGNETIC_LATITUDE'])
        variables['MAOB_S_EQ_MLON'] = np.array(dataset.variables['MODEL_SOUTH_GEOMAGNETIC_LONGITUDE'])
        variables['MAOB_S_EQ_MLT'] = np.array(dataset.variables['MODEL_SOUTH_MAGNETIC_LOCAL_TIME'])

        variables['MAOB_S_PL_MLAT'] = np.array(dataset.variables['MODEL_SOUTH_POLAR_GEOMAGNETIC_LATITUDE'])
        variables['MAOB_S_PL_MLON'] = np.array(dataset.variables['MODEL_SOUTH_POLAR_GEOMAGNETIC_LONGITUDE'])
        variables['MAOB_S_PL_MLT'] = np.array(dataset.variables['MODEL_SOUTH_POLAR_MAGNETIC_LOCAL_TIME'])

        dataset.close()

        self.variables = variables


if __name__ == "__main__":
    pass


    # if hasattr(readObj, 'pole'):
    #    readObj.filter_data_pole(boundinglat = 25)