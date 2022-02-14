import numpy as np
import netCDF4 as nc
import datetime
import pathlib
import cftime

from sscws.sscws import SscWs
from sscws.coordinates import CoordinateComponent, CoordinateSystem, \
    SurfaceGeographicCoordinates

import geospacelab.cs as gsl_cs
import geospacelab.toolbox.utilities.pydatetime as dttool


def sat_orbits(
        dt_fr: datetime.datetime = None, dt_to: datetime.datetime = None,
        sat_id='f18', to_AACGM='on',
):
    # initialization
    ssc = SscWs()
    sat = 'dmsp' + sat_id.lower()
    dt_fr_str = dt_fr.strftime('%Y-%m-%dT%H:%M:%SZ')
    dt_to_str = dt_to.strftime('%Y-%m-%dT%H:%M:%SZ')

    result = ssc.get_locations(
        [sat],
        [dt_fr_str, dt_to_str], [CoordinateSystem.GEO, CoordinateSystem.GSE]
    )

    if not list(result['Data']):
        return None

    data = result['Data'][0]
    coords = data['Coordinates'][0]
    coords_gse = data['Coordinates'][1]
    dts = data['Time']

    coords_in = {'x': coords['X'] / 6371.2, 'y': coords['Y'] / 6371.2, 'z': coords['Z'] / 6371.2}
    cs_car = gsl_cs.GEOCCartesian(coords=coords_in, ut=dts)
    cs_sph = cs_car.to_spherical()
    orbits = {
        'GEO_LAT': cs_sph['lat'],
        'GEO_LON': cs_sph['lon'],
        'GEO_ALT': cs_sph['height'],
        'GEO_X': coords['X'],
        'GEO_Y': coords['Y'],
        'GEO_Z': coords['Z'],
        'GSE_X': coords_gse['X'],
        'GSE_Y': coords_gse['Y'],
        'GSE_Z': coords_gse['Z'],
        'DATETIME': dts,
    }

    if to_AACGM:
        cs_aacgm = cs_sph.to_AACGM(append_mlt=True)
        orbits.update(
            **{
                'AACGM_LAT': cs_aacgm['lat'],
                'AACGM_LON': cs_aacgm['lon'],
                'AACGM_MLT': cs_aacgm['mlt'],
            }
        )
    return orbits


def check_option_sat_time():
    sat_ids = ['f11', 'f12', 'f13', 'f14', 'f15', 'f16', 'f17', 'f18', 'f19']
    # sat_ids = ['f15']

    dt_fr = datetime.datetime(2000, 1, 1, 0)
    dt_to = datetime.datetime(2021, 12, 31, 12, 59, 59)

    diff_months = (dt_to.year - dt_fr.year) * 12 + (dt_to.month - dt_fr.month) % 12

    for nm in range(diff_months + 1):
        thismonth = dttool.get_next_n_months(dt_fr, nm)
        thismonthend = dttool.get_last_day_of_month(thismonth, end=True)

        for sat_id in sat_ids:
            orbits = dmsp_orbits(dt_fr=thismonth, dt_to=thismonthend, sat_id=sat_id, to_AACGM='on')
            if orbits is None:
                continue
            fp = pathlib.Path(__file__).resolve().parent.resolve() / f'SSCWS_orbits_DMSP-{sat_id.upper()}_{thismonth.strftime("%Y%m")}_1min.nc'
            fp.parent.resolve().mkdir(parents=True, exist_ok=True)
            fnc = nc.Dataset(fp, 'w')
            fnc.createDimension('UNIX_TIME', orbits['DATETIME'].shape[0])

            fnc.title = f"DMSP-{sat_id.upper()} orbits from {thismonth.strftime('%Y%m%dT%H%M%S')} to {thismonthend.strftime('%Y%m%dT%H%M%S')}"
            time = fnc.createVariable('UNIX_TIME', np.float64, ('UNIX_TIME',))
            time.units = 'seconds since 1970-01-01 00:00:00.0'
            geo_lat = fnc.createVariable('GEO_LAT', np.float32, ('UNIX_TIME',))
            geo_lon = fnc.createVariable('GEO_LON', np.float32, ('UNIX_TIME',))
            geo_alt = fnc.createVariable('GEO_ALT', np.float32, ('UNIX_TIME',))
            geo_x = fnc.createVariable('GEO_X', np.float32, ('UNIX_TIME',))
            geo_y = fnc.createVariable('GEO_Y', np.float32, ('UNIX_TIME',))
            geo_z = fnc.createVariable('GEO_Z', np.float32, ('UNIX_TIME',))
            gse_x = fnc.createVariable('GSE_X', np.float32, ('UNIX_TIME',))
            gse_y = fnc.createVariable('GSE_Y', np.float32, ('UNIX_TIME',))
            gse_z = fnc.createVariable('GSE_Z', np.float32, ('UNIX_TIME',))
            aa_lat = fnc.createVariable('AACGM_LAT', np.float32, ('UNIX_TIME',))
            aa_lon = fnc.createVariable('AACGM_LON', np.float32, ('UNIX_TIME',))
            aa_mlt = fnc.createVariable('AACGM_MLT', np.float32, ('UNIX_TIME',))

            time_array = np.array(cftime.date2num(orbits['DATETIME'].flatten(), units='seconds since 1970-01-01 00:00:00.0'))
            time[::] = time_array[::]
            geo_lat[::] = orbits['GEO_LAT'][::]
            geo_lon[::] = orbits['GEO_LON'][::]
            geo_alt[::] = orbits['GEO_ALT'][::]
            geo_x[::] = orbits['GEO_X'][::]
            geo_y[::] = orbits['GEO_Y'][::]
            geo_z[::] = orbits['GEO_Z'][::]
            gse_x[::] = orbits['GSE_X'][::]
            gse_y[::] = orbits['GSE_Y'][::]
            gse_z[::] = orbits['GSE_Z'][::]
            aa_lat[::] = orbits['AACGM_LAT'][::]
            aa_lon[::] = orbits['AACGM_LON'][::]
            aa_mlt[::] = orbits['AACGM_MLT'][::]

            print(f"DMSP-{sat_id.upper()} orbits from {thismonth.strftime('%Y%m%dT%H%M%S')} to {thismonthend.strftime('%Y%m%dT%H%M%S')}")
            fnc.close()


def test():
    sat_id = 'f18'
    dt_fr = datetime.datetime(2010, 1, 1, 0)
    dt_to = datetime.datetime(2022, 1, 1, 12, 59, 59)
    orbits = dmsp_orbits(dt_fr=dt_fr, dt_to=dt_to, sat_id=sat_id)

    import matplotlib.pyplot as plt
    plt.plot(orbits['DATETIME'], orbits['GEO_LAT'])


if __name__ == "__main__":
    check_option_sat_time()
