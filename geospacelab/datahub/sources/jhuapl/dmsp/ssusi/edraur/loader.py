import netCDF4
import datetime
import numpy as np
import geospacelab.toolbox.utilities.pydatetime as dttool


class Loader(object):

    def __init__(self, file_path, file_type='edr-aur', pole='S'):

        self.variables = {}
        self.metadata = {}
        self.file_path = file_path
        self.file_type = file_type
        self.pole = pole

        self.load_data()

    def load_data(self):
        dataset = netCDF4.Dataset(self.file_path)
        variables = {}

        if self.pole == 'N':
            pole = self.pole
            pole_str = 'NORTH'
        elif self.pole == 'S':
            pole = self.pole
            pole_str = 'SOUTH'
        else:
            raise ValueError
        # Time and Position
        # sectime = int(np.array(dataset.variables['TIME']).flatten()[0])
        # doy = int(np.array(dataset.variables['DOY']).flatten()[0])
        # year = int(np.array(dataset.variables['YEAR']).flatten()[0])
        # dt0 = dttool.convert_doy_to_datetime(year, doy)
        starting_time = datetime.datetime.strptime(dataset.STARTING_TIME, "%Y%j%H%M%S")
        variables['STARTING_TIME'] = starting_time
        stopping_time = datetime.datetime.strptime(dataset.STOPPING_TIME, "%Y%j%H%M%S")
        variables['STOPPING_TIME'] = stopping_time
        dt0 = dttool.get_start_of_the_day(starting_time)

        variables['SC_LAT'] = np.array(dataset.variables['LATITUDE'])
        variables['SC_LON'] = np.array(dataset.variables['LONGITUDE'])
        variables['SC_ALT'] = np.array(dataset.variables['ALTITUDE'])

        # Auroral map, #colors: 0: '1216', 1: '1304', 2: '1356', 3: 'LBHS', 4: 'LBHL'.
        variables['EMISSION_SPECTRA'] = ['1216', '1304', '1356', 'LBHS', 'LBHL']
        disk_aur = np.array(dataset.variables['DISK_RADIANCEDATA_INTENSITY_' + pole_str])
        disk_aur[disk_aur <= 0] = np.nan
        variables['GRID_AUR_1216'] = disk_aur[0, ::]
        variables['GRID_AUR_1304'] = disk_aur[1, ::]
        variables['GRID_AUR_1356'] = disk_aur[2, ::]
        variables['GRID_AUR_LBHS'] = disk_aur[3, ::]
        variables['GRID_AUR_LBHL'] = disk_aur[4, ::]

        variables['GRID_MLAT'] = np.array(dataset.variables['LATITUDE_GEOMAGNETIC_GRID_MAP'])
        variables['GRID_MLON'] = np.array(
            dataset.variables['LONGITUDE_GEOMAGNETIC_' + pole_str + '_GRID_MAP'])
        variables['GRID_MLT'] = np.array(dataset.variables['MLT_GRID_MAP'])
        if self.pole == 'S':
            variables['GRID_MLAT'] = - variables['GRID_MLAT']
        variables['GRID_UT'] = np.array(dataset.variables['UT_' + pole])
        lat = np.array(variables['GRID_MLAT'])
        ut = np.array(variables['GRID_UT'])
        lat = np.where(ut == 0, np.nan, lat)
        if self.pole == 'N':
            ind_mid_t = np.where(lat == np.nanmax(lat.flatten()))
        else:
            ind_mid_t = np.where(lat == np.nanmin(lat.flatten()))
        sectime0 = variables['GRID_UT'][ind_mid_t][0] * 3600
        variables['DATETIME'] = dt0 + datetime.timedelta(seconds=int(sectime0))

        # Auroral oval boundary
        variables['AOB_EQ_MLAT'] = np.array(dataset.variables[pole_str + '_GEOMAGNETIC_LATITUDE'])
        variables['AOB_EQ_MLON'] = np.array(dataset.variables[pole_str + '_GEOMAGNETIC_LONGITUDE'])
        variables['AOB_EQ_MLT'] = np.array(dataset.variables[pole_str + '_MAGNETIC_LOCAL_TIME'])

        variables['AOB_PL_MLAT'] = np.array(dataset.variables[pole_str + '_POLAR_GEOMAGNETIC_LATITUDE'])
        variables['AOB_PL_MLON'] = np.array(dataset.variables[pole_str + '_POLAR_GEOMAGNETIC_LONGITUDE'])
        variables['AOB_PL_MLT'] = np.array(dataset.variables[pole_str + '_POLAR_MAGNETIC_LOCAL_TIME'])

        variables['MAOB_EQ_MLAT'] = np.array(dataset.variables['MODEL_' + pole_str + '_GEOMAGNETIC_LATITUDE'])
        variables['MAOB_EQ_MLON'] = np.array(dataset.variables['MODEL_' + pole_str + '_GEOMAGNETIC_LONGITUDE'])
        variables['MAOB_EQ_MLT'] = np.array(dataset.variables['MODEL_' + pole_str + '_MAGNETIC_LOCAL_TIME'])

        variables['MAOB_PL_MLAT'] = np.array(dataset.variables['MODEL_' + pole_str + '_POLAR_GEOMAGNETIC_LATITUDE'])
        variables['MAOB_PL_MLON'] = np.array(dataset.variables['MODEL_' + pole_str + '_POLAR_GEOMAGNETIC_LONGITUDE'])
        variables['MAOB_PL_MLT'] = np.array(dataset.variables['MODEL_' + pole_str + '_POLAR_MAGNETIC_LOCAL_TIME'])

        dataset.close()

        self.variables = variables


if __name__ == "__main__":
    import pathlib
    fp = pathlib.Path('/Users/lcai/Geospacelab/Data/JHUAPL/DMSP/SSUSI/f17/20151205/' +
                      'PS.APL_V0105S027CE0008_SC.U_DI.A_GP.F17-SSUSI_PA.APL-EDR-AURORA_DD.20151205_SN.46863-00_DF.NC')
    loader = Loader(file_path=fp)


    # if hasattr(readObj, 'pole'):
    #    readObj.filter_data_pole(boundinglat = 25)