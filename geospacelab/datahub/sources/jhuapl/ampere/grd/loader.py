# Licensed under the BSD 3-Clause License
# Copyright (C) 2021 GeospaceLAB (geospacelab)
# Author: Lei Cai, Space Physics and Astronomy, University of Oulu

import netCDF4
import datetime
import numpy as np
import geospacelab.toolbox.utilities.pydatetime as dttool


class Loader(object):

    def __init__(self, file_path, file_type='grd', pole='N'):

        self.variables = {}
        self.metadata = {}
        self.file_path = file_path
        self.file_type = file_type
        self.pole = pole

        self.load_data()

    def load_data(self):
        dataset = netCDF4.Dataset(self.file_path)
        variables = {}

        dts = np.array(
            datetime.datetime(yy, mm, dd) + datetime.timedelta(hours=np.round)
        )


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

# dataset.variables
# {'npnt': <class 'netCDF4._netCDF4.Variable'>
# int16 npnt(nRec)
#     comment: number of observations in record.
# unlimited dimensions:
# current shape = (720,)
# filling on, default _FillValue of -32767 used, 'year': <class 'netCDF4._netCDF4.Variable'>
# int16 year(nRec)
#     comment: year.
# unlimited dimensions:
# current shape = (720,)
# filling on, default _FillValue of -32767 used, 'doy': <class 'netCDF4._netCDF4.Variable'>
# int16 doy(nRec)
#     comment: day of year (1-366).
# unlimited dimensions:
# current shape = (720,)
# filling on, default _FillValue of -32767 used, 'time': <class 'netCDF4._netCDF4.Variable'>
# float32 time(nRec)
#     comment: time in units of fractional hours of day.
# unlimited dimensions:
# current shape = (720,)
# filling on, default _FillValue of 9.969209968386869e+36 used, 'avgint': <class 'netCDF4._netCDF4.Variable'>
# int16 avgint(nRec)
#     comment: time averaging window length in seconds.
# unlimited dimensions:
# current shape = (720,)
# filling on, default _FillValue of -32767 used, 'kmax': <class 'netCDF4._netCDF4.Variable'>
# int16 kmax(nRec)
#     comment: latitude order of fit.
# unlimited dimensions:
# current shape = (720,)
# filling on, default _FillValue of -32767 used, 'mmax': <class 'netCDF4._netCDF4.Variable'>
# int16 mmax(nRec)
#     comment: longitude order of fit.
# unlimited dimensions:
# current shape = (720,)
# filling on, default _FillValue of -32767 used, 'res_deg': <class 'netCDF4._netCDF4.Variable'>
# float32 res_deg(nRec)
#     comment: grid latitude resolution in degrees.
# unlimited dimensions:
# current shape = (720,)
# filling on, default _FillValue of 9.969209968386869e+36 used, 'nLatGrid': <class 'netCDF4._netCDF4.Variable'>
# int16 nLatGrid(nRec)
#     comment: number of latitude points in grid.
# unlimited dimensions:
# current shape = (720,)
# filling on, default _FillValue of -32767 used, 'nLonGrid': <class 'netCDF4._netCDF4.Variable'>
# int16 nLonGrid(nRec)
#     comment: number of longitude points in grid.
# unlimited dimensions:
# current shape = (720,)
# filling on, default _FillValue of -32767 used, 'cLat_deg': <class 'netCDF4._netCDF4.Variable'>
# float32 cLat_deg(nRec, nObs)
#     comment: co-latitude in AACGM coordinates in degrees.
# unlimited dimensions:
# current shape = (720, 1200)
# filling on, default _FillValue of 9.969209968386869e+36 used, 'mlt_hr': <class 'netCDF4._netCDF4.Variable'>
# float32 mlt_hr(nRec, nObs)
#     comment: AACGM Magnetic Local Time (MLT) in hours.
# unlimited dimensions:
# current shape = (720, 1200)
# filling on, default _FillValue of 9.969209968386869e+36 used, 'geo_cLat_deg': <class 'netCDF4._netCDF4.Variable'>
# float32 geo_cLat_deg(nRec, nObs)
#     comment: co-latitude in GEO coordinates in degrees.
# unlimited dimensions:
# current shape = (720, 1200)
# filling on, default _FillValue of 9.969209968386869e+36 used, 'geo_lon_deg': <class 'netCDF4._netCDF4.Variable'>
# float32 geo_lon_deg(nRec, nObs)
#     comment: longitude in GEO coordinates in degrees.
# unlimited dimensions:
# current shape = (720, 1200)
# filling on, default _FillValue of 9.969209968386869e+36 used, 'R': <class 'netCDF4._netCDF4.Variable'>
# float32 R(nRec, nObs)
#     comment: Radius from center of the Earth in kilometers.
# unlimited dimensions:
# current shape = (720, 1200)
# filling on, default _FillValue of 9.969209968386869e+36 used, 'pos_geo': <class 'netCDF4._netCDF4.Variable'>
# float32 pos_geo(nRec, nObs, vComp)
#     comment: vehicle position in GEO coordinates in units of kilometers.
# unlimited dimensions:
# current shape = (720, 1200, 3)
# filling on, default _FillValue of 9.969209968386869e+36 used, 'db_R': <class 'netCDF4._netCDF4.Variable'>
# float64 db_R(nRec, nObs)
#     comment: magnetic field perturbation parallel to GEO radial direction in units of nano-Tesla.
# unlimited dimensions:
# current shape = (720, 1200)
# filling on, default _FillValue of 9.969209968386869e+36 used, 'db_T': <class 'netCDF4._netCDF4.Variable'>
# float64 db_T(nRec, nObs)
#     comment: magnetic field perturbation parallel to GEO northward direction in units of nano-Tesla.
# unlimited dimensions:
# current shape = (720, 1200)
# filling on, default _FillValue of 9.969209968386869e+36 used, 'db_P': <class 'netCDF4._netCDF4.Variable'>
# float64 db_P(nRec, nObs)
#     comment: magnetic field perturbation parallel to GEO eastward direction in units of nano-Tesla.
# unlimited dimensions:
# current shape = (720, 1200)
# filling on, default _FillValue of 9.969209968386869e+36 used, 'db_geo': <class 'netCDF4._netCDF4.Variable'>
# float64 db_geo(nRec, nObs, vComp)
# unlimited dimensions:
# current shape = (720, 1200, 3)
# filling on, default _FillValue of 9.969209968386869e+36 used, 'jPar': <class 'netCDF4._netCDF4.Variable'>
# float64 jPar(nRec, nObs)
#     comment: Radial current density [muA/m^2]
# unlimited dimensions:
# current shape = (720, 1200)
# filling on, default _FillValue of 9.969209968386869e+36 used, 'db_Th_Th': <class 'netCDF4._netCDF4.Variable'>
# float64 db_Th_Th(nRec, nObs)
#     comment: Magnetic field perturbation parallel to the AACGM northward direction in units of nano-Tesla.
# unlimited dimensions:
# current shape = (720, 1200)
# filling on, default _FillValue of 9.969209968386869e+36 used, 'db_Ph_Th': <class 'netCDF4._netCDF4.Variable'>
# float64 db_Ph_Th(nRec, nObs)
#     comment: Magnetic field perturbation perpendicular to the AACGM northward direction in units of nano-Tesla.
# unlimited dimensions:
# current shape = (720, 1200)
# filling on, default _FillValue of 9.969209968386869e+36 used, 'db_Th_Ph': <class 'netCDF4._netCDF4.Variable'>
# float64 db_Th_Ph(nRec, nObs)
#     comment: Magnetic field perturbation perpendicular  to the AACGM eastward direction in units of nano-Tesla.
# unlimited dimensions:
# current shape = (720, 1200)
# filling on, default _FillValue of 9.969209968386869e+36 used, 'db_Ph_Ph': <class 'netCDF4._netCDF4.Variable'>
# float64 db_Ph_Ph(nRec, nObs)
#     comment: Magnetic field perturbation parallel to the AACGM eastward direction in units of nano-Tesla.
# unlimited dimensions:
# current shape = (720, 1200)
# filling on, default _FillValue of 9.969209968386869e+36 used, 'del_db_R': <class 'netCDF4._netCDF4.Variable'>
# float64 del_db_R(nRec, nObs)
#     comment: magnetic field residual parallel to GEO radial direction in units of nano-Tesla.
# unlimited dimensions:
# current shape = (720, 1200)
# filling on, default _FillValue of 9.969209968386869e+36 used, 'del_db_T': <class 'netCDF4._netCDF4.Variable'>
# float64 del_db_T(nRec, nObs)
#     comment: magnetic field residual parallel to GEO northward direction in units of nano-Tesla.
# unlimited dimensions:
# current shape = (720, 1200)
# filling on, default _FillValue of 9.969209968386869e+36 used, 'del_db_P': <class 'netCDF4._netCDF4.Variable'>
# float64 del_db_P(nRec, nObs)
#     comment: magnetic field residual parallel to GEO eastward direction in units of nano-Tesla.
# unlimited dimensions:
# current shape = (720, 1200)
# filling on, default _FillValue of 9.969209968386869e+36 used, 'del_db_geo': <class 'netCDF4._netCDF4.Variable'>
# float64 del_db_geo(nRec, nObs, vComp)
#     comment: magnetic field residuals in GEO coordinates in units of nano-Tesla.
# unlimited dimensions:
# current shape = (720, 1200, 3)
# filling on, default _FillValue of 9.969209968386869e+36 used, 'del_jPar': <class 'netCDF4._netCDF4.Variable'>
# float64 del_jPar(nRec, nObs)
#     comment: Radial current density residual [muA/m^2]
# unlimited dimensions:
# current shape = (720, 1200)
# filling on, default _FillValue of 9.969209968386869e+36 used, 'del_db_Th_Th': <class 'netCDF4._netCDF4.Variable'>
# float64 del_db_Th_Th(nRec, nObs)
#     comment: Magnetic field residual parallel to the AACGM northward direction in units of nano-Tesla.
# unlimited dimensions:
# current shape = (720, 1200)
# filling on, default _FillValue of 9.969209968386869e+36 used, 'del_db_Ph_Th': <class 'netCDF4._netCDF4.Variable'>
# float64 del_db_Ph_Th(nRec, nObs)
#     comment: Magnetic field residual perpendicular to the AACGM northward direction in units of nano-Tesla.
# unlimited dimensions:
# current shape = (720, 1200)
# filling on, default _FillValue of 9.969209968386869e+36 used, 'del_db_Th_Ph': <class 'netCDF4._netCDF4.Variable'>
# float64 del_db_Th_Ph(nRec, nObs)
#     comment: Magnetic field residual perpendicular  to the AACGM eastward direction in units of nano-Tesla.
# unlimited dimensions:
# current shape = (720, 1200)
# filling on, default _FillValue of 9.969209968386869e+36 used, 'del_db_Ph_Ph': <class 'netCDF4._netCDF4.Variable'>
# float64 del_db_Ph_Ph(nRec, nObs)
#     comment: Magnetic field residual parallel to the AACGM eastward direction in units of nano-Tesla.
# unlimited dimensions:
# current shape = (720, 1200)
# filling on, default _FillValue of 9.969209968386869e+36 used}
