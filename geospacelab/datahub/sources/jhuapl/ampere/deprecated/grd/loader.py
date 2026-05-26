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

        year = dataset.variables['year'][::][0]
        start_of_year = datetime.datetime(year, 1, 1) 
        hhs = [int(np.floor(t)) for t in dataset.variables['time'][::]]
        mms = [int(np.round((t-hh)*60)) for t, hh in zip(dataset.variables['time'][::], hhs)] 
        sss = [int(np.round((t-hh - mm/60)*60)) for t, hh, mm in zip(dataset.variables['time'][::], hhs, mms)]  
        dts = np.array([
            start_of_year + datetime.timedelta(days=int(doy)-1, hours=hh, minutes=mm, seconds=ss)
            for doy, hh, mm, ss in zip(
                dataset.variables['doy'][::],
                hhs,
                mms,
                sss,
            )
        ])

        ntime = dts.shape[0]
        nlon = dataset.variables['nLonGrid'][::][0]
        nlat = dataset.variables['nLatGrid'][::][0]
        colat = dataset.variables['cLat_deg'][::].reshape((ntime, nlon, nlat))

        mlt = dataset.variables['mlt_hr'][::].reshape((ntime, nlon, nlat))

        Jr = dataset.variables['jPar'][::].reshape(ntime, nlon, nlat)

        variables['DATETIME'] = dts[:, np.newaxis]

        variables['GRID_MLAT'] = np.array(90. - colat)
        variables['GRID_MLT'] = np.array(mlt)

        variables['GRID_Jr'] = np.array(Jr)

        dataset.close()

        self.variables = variables


cdf_var_names = [
    'npnt', 'year', 'doy', 'time', 'avgint',
    'kmax', 'mmax', 'res_deg', 'nLatGrid', 'nLonGrid',
    'cLat_deg', 'mlt_hr', 'geo_cLat_deg', 'geo_lon_deg', 'R',
    'pos_geo', 'db_R', 'db_T', 'db_P', 'db_geo',
    'jPar', 'db_Th_Th', 'db_Ph_Th', 'db_Th_Ph', 'db_Ph_Ph',
    'del_db_R', 'del_db_T', 'del_db_P', 'del_db_geo', 'del_jPar',
    'del_db_Th_Th', 'del_db_Ph_Th', 'del_db_Th_Ph', 'del_db_Ph_Ph'
]




if __name__ == "__main__":
    import pathlib
    fp = pathlib.Path('/home/lei/git-repos/geospacelab/geospacelab/datahub/sources/jhuapl/ampere/20160314/AMPERE_GRD_20160314T0000_20160314T0100_N.nc')
    # fp = "/home/lei/north.nc"
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
