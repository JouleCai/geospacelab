# Licensed under the BSD 3-Clause License
# Copyright (C) 2021 GeospaceLab (geospacelab)
# Author: Lei Cai, Space Physics and Astronomy, University of Oulu

import netCDF4 as nc
import datetime
import numpy as np
import pathlib
import re
import cftime
import geospacelab.toolbox.utilities.pydatetime as dttool
import geospacelab.toolbox.utilities.pylogging as mylog


class Loader(object):

    def __init__(self, file_path, file_ext='nc', pole='N', append_support_data=True):

        self.variables = {}
        self.metadata = {}
        self.file_path = pathlib.Path(file_path)
        self.file_ext = file_ext
        self.pole = pole
        self.append_support_data = append_support_data

        self.load_data()

    def load_data(self):
        if self.file_path.suffix.strip('.') in ['dat', 'txt'] or self.file_ext == 'ascii':
            file_nc = pathlib.Path(self.file_path.with_suffix('.nc'))
            if not file_nc.is_file():
                self.save_to_nc(self.file_path)
            self.file_ext = 'nc'
            self.file_path = file_nc
            
        if self.file_ext == 'nc':
            self.load_data_nc()
    
    def load_data_nc(self):

        dataset = nc.Dataset(self.file_path)
        variables = {}

        unix_time = dataset.variables['UNIX_TIME'][::]
        dts = dttool.convert_unix_time_to_datetime_cftime(unix_time)
        ntime = dts.size
        variables['DATETIME'] = np.reshape(dts, (ntime, 1))

        variables['GRID_MLAT'] = np.array(dataset.variables['MLAT'][::])
        variables['GRID_MLON'] = np.array(dataset.variables['MLON'][::])
        variables['GRID_MLT'] = np.array(dataset.variables['MLT'][::])
        variables['GRID_E_E'] = np.array(dataset.variables['E_E'][::])
        variables['GRID_E_E'] = np.array(dataset.variables['E_E'][::])
        variables['GRID_E_N'] = np.array(dataset.variables['E_N'][::])
        variables['GRID_v_i_E'] = np.array(dataset.variables['v_i_E'][::])
        variables['GRID_v_i_N'] = np.array(dataset.variables['v_i_E'][::])
        variables['GRID_phi'] = np.array(dataset.variables['phi'][::])

        if self.append_support_data:
            var_names_append = [
                'VCNUM', 'IMF_MODEL', 'CLOCK_ANGLE', 'DIP_TILT', 'E_SW',
                'SD_MODEL', 'FIT_ORDER', 'B_x_OMNI', 'B_y_OMNI', 'B_z_OMNI',
                'phi_CPCP', 'phi_MAX', 'phi_MIN'
            ]
            for var_name in var_names_append:
                variables[var_name] = np.array(dataset.variables[var_name][::]).reshape((ntime, 1))


        dataset.close()

        self.variables = variables

    def save_to_nc(self, file_path):
        if not file_path.is_file():
            raise FileExistsError
        with open(file_path, 'r') as f:
            text = f.read()

            # results = re.findall(
            #     r'^\s*(\d+)\s*\[(\d+),(\d+)]\s*([-\d.]+)\s*' +
            #     r'([-\d.]+)\s*([-\d.]+)\s*([-\d.]+)\s*([-\d.]+)\s*([-\d.]+)\s*([-\d.]+)\s*' +
            #     r'([\S]+)',
            #     text,
            #     re.M
            # )
            results = re.findall(
                r'^\s*(\d+)\s*\[(\d+),(\d+)]\s*([\S]+)\s*([\S]+)\s*([\S]+)\s*([\S]+)\s*([\S]+)\s*([\S]+)\s*([\S]+)\s*([\S]+)',
                text,
                re.M
            )
            results = list(zip(*results))
            nlat = 40
            nlon = 180
            ntime = len(results[0]) / nlon / nlat
            if ntime != int(ntime):
                raise ValueError
            ntime = int(ntime)
            mlat_arr = np.array(results[3]).reshape([ntime, nlat, nlon], order='C').transpose((0, 2, 1)).astype(np.float32)
            mlon_arr = np.array(results[4]).reshape([ntime, nlat, nlon], order='C').transpose((0, 2, 1)).astype(np.float32)
            EF_N_arr = np.array(results[5]).reshape([ntime, nlat, nlon], order='C').transpose((0, 2, 1)).astype(np.float32)
            EF_E_arr = np.array(results[6]).reshape([ntime, nlat, nlon], order='C').transpose((0, 2, 1)).astype(np.float32)
            v_N_arr = np.array(results[7]).reshape([ntime, nlat, nlon], order='C').transpose((0, 2, 1)).astype(np.float32)
            v_E_arr = np.array(results[8]).reshape([ntime, nlat, nlon], order='C').transpose((0, 2, 1)).astype(np.float32)
            phi_arr = np.array(results[9]).reshape([ntime, nlat, nlon], order='C').transpose((0, 2, 1)).astype(np.float32)

            dts = np.array(results[10])[::nlon * nlat]
            dts = [datetime.datetime.strptime(dtstr, "%Y-%m-%d/%H:%M:%S") for dtstr in dts]
            time_array = np.array(cftime.date2num(dts, units='seconds since 1970-01-01 00:00:00.0'))

            import aacgmv2
            mlt_arr = np.empty_like(mlat_arr)
            for i in range(ntime):
                mlt1 = aacgmv2.convert_mlt(mlon_arr[i].flatten(), dts[i]).reshape((nlon, nlat))
                mlt_arr[i, ::] = mlt1[::]

            fp = pathlib.Path(file_path.with_suffix('.nc'))
            fp.parent.resolve().mkdir(parents=True, exist_ok=True)
            fnc = nc.Dataset(fp, 'w')
            fnc.createDimension('UNIX_TIME', ntime)
            fnc.createDimension('MLAT', nlat)
            fnc.createDimension('MLON', nlon)

            fnc.title = "SuperDARN Potential maps"

            time = fnc.createVariable('UNIX_TIME', np.float64, ('UNIX_TIME',))
            time.units = 'seconds since 1970-01-01 00:00:00.0'
            time[::] = time_array[::]

            mlat = fnc.createVariable('MLAT', np.float32, ('UNIX_TIME', 'MLON', 'MLAT'))
            mlat[::] = mlat_arr[::]
            mlon = fnc.createVariable('MLON', np.float32, ('UNIX_TIME', 'MLON', 'MLAT'))
            mlon[::] = mlon_arr[::]
            mlt = fnc.createVariable('MLT', np.float32, ('UNIX_TIME', 'MLON', 'MLAT'))
            mlt[::] = mlt_arr[::]
            EF_N = fnc.createVariable('E_N', np.float32, ('UNIX_TIME', 'MLON', 'MLAT'))
            EF_N[::] = EF_N_arr[::]
            EF_E = fnc.createVariable('E_E', np.float32, ('UNIX_TIME', 'MLON', 'MLAT'))
            EF_E[::] = EF_E_arr[::]
            v_N = fnc.createVariable('v_i_N', np.float32, ('UNIX_TIME', 'MLON', 'MLAT'))
            v_N[::] = v_N_arr[::]
            v_E = fnc.createVariable('v_i_E', np.float32, ('UNIX_TIME', 'MLON', 'MLAT'))
            v_E[::] = v_E_arr[::]
            phi = fnc.createVariable('phi', np.float32, ('UNIX_TIME', 'MLON', 'MLAT'))
            phi[::] = phi_arr[::]

            # Support data such as IMF, VCNUM, potential drop
            results = re.findall(
                r'^>[A-Za-z ]*\(VCNUM\):\s*([\d]+)',
                text,
                re.M
            )
            vcnum_arr = np.array(results, dtype=np.int32).reshape([ntime, 1], order='C')
            vcnum = fnc.createVariable('VCNUM', 'i4', ('UNIX_TIME',))
            vcnum[::] = vcnum_arr[::]

            results = re.findall(
                r'^>\s*IMF Model: ([\S]+) Bang\s*([\S]+) deg\., '
                + r'Esw\s*([\S]+) mV/m, tilt\s*([\S]+) deg\.,\s*([A-Za-z0-9]+), Fit Order:\s*([\d]+)',
                text,
                re.M
            )
            results = list(zip(*results))
            imf_model_arr = np.array(results[0], dtype=object).reshape([ntime, 1], order='C')
            imf_model = fnc.createVariable('IMF_MODEL', 'S8', ('UNIX_TIME',))
            imf_model[::] = imf_model_arr[::]
            B_angle_arr = np.array(results[1], dtype=np.float32).reshape([ntime, 1], order='C')
            B_angle = fnc.createVariable('CLOCK_ANGLE', np.float32, ('UNIX_TIME',))
            B_angle[::] = B_angle_arr[::]
            E_sw_arr = np.array(results[2], dtype=np.float32).reshape([ntime, 1], order='C')
            E_sw = fnc.createVariable('E_SW', np.float32, ('UNIX_TIME',))
            E_sw[::] = E_sw_arr[::]
            dipole_tilt_arr = np.array(results[3], dtype=np.float32).reshape([ntime, 1], order='C')
            dipole_tilt = fnc.createVariable('DIP_TILT', np.float32, ('UNIX_TIME',))
            dipole_tilt[::] = dipole_tilt_arr[::]
            SD_model_arr = np.array(results[4], dtype=object).reshape([ntime, 1], order='C')
            SD_model = fnc.createVariable('SD_MODEL', 'S8', ('UNIX_TIME',))
            SD_model[::] = SD_model_arr[::]
            fit_order_arr = np.array(results[5], dtype=np.int32).reshape([ntime, 1], order='C')
            fit_order = fnc.createVariable('FIT_ORDER', 'i4', ('UNIX_TIME',))
            fit_order[::] = fit_order_arr[::]

            results = re.findall(
                r'^> OMNI IMF:\s*Bx=([\S]+) nT,\s*By=([\S]+) nT,\s*Bz=([\S]+) nT',
                text,
                re.M
            )
            results = list(zip(*results))
            B_x_OMNI_arr = np.array(results[0], dtype=np.float32).reshape([ntime, 1], order='C')
            B_x_OMNI = fnc.createVariable('B_x_OMNI', np.float32, ('UNIX_TIME',))
            B_x_OMNI[::] = B_x_OMNI_arr[::]
            B_y_OMNI_arr = np.array(results[1], dtype=np.float32).reshape([ntime, 1], order='C')
            B_y_OMNI = fnc.createVariable('B_y_OMNI', np.float32, ('UNIX_TIME',))
            B_y_OMNI[::] = B_y_OMNI_arr[::]
            B_z_OMNI_arr = np.array(results[2], dtype=np.float32).reshape([ntime, 1], order='C')
            B_z_OMNI = fnc.createVariable('B_z_OMNI', np.float32, ('UNIX_TIME',))
            B_z_OMNI[::] = B_z_OMNI_arr[::]

            # > Potential: Drop = 33 kV, Min = -19 kV, Max = 14 kV
            results = re.findall(
                r'^> Potential:\s*Drop=([\S]+) kV,\s*Min=([\S]+) kV,\s*Max=([\S]+) kV',
                text,
                re.M
            )
            results = list(zip(*results))
            phi_CPCP_arr = np.array(results[0], dtype=np.float32).reshape([ntime, 1], order='C')
            phi_CPCP = fnc.createVariable('phi_CPCP', np.float32, ('UNIX_TIME',))
            phi_CPCP[::] = phi_CPCP_arr[::]
            phi_max_arr = np.array(results[1], dtype=np.float32).reshape([ntime, 1], order='C')
            phi_max = fnc.createVariable('phi_MAX', np.float32, ('UNIX_TIME',))
            phi_max[::] = phi_max_arr[::]
            phi_min_arr = np.array(results[2], dtype=np.float32).reshape([ntime, 1], order='C')
            phi_min = fnc.createVariable('phi_MIN', np.float32, ('UNIX_TIME',))
            phi_min[::] = phi_min_arr[::]

            print('From {} to {}.'.format(
                datetime.datetime.utcfromtimestamp(time_array[0]),
                datetime.datetime.utcfromtimestamp(time_array[-1]))
            )
            mylog.StreamLogger.info(
                "The requested SuperDARN map potential data has been saved in the file {}.".format(fp))
            fnc.close()

if __name__ == "__main__":
    import pathlib
    fp = pathlib.Path('/home/lei/afys-data/SuperDARN/PotentialMap/2016/SuperDARN_POTMAP_2min_20160315_N.nc')
    loader = Loader(file_path=fp)


    # if hasattr(readObj, 'pole'):
    #    readObj.filter_data_pole(boundinglat = 25)