import re
import pathlib
import datetime
import numpy as np

import geospacelab.datahub.sources.supermag.supermag_api as smapi
from geospacelab.config import prf
import geospacelab.toolbox.utilities.pylogging as mylog


def list_available_sites(dt_fr=None, dt_to=None, extent=None, user_id=None):
    if user_id is None:
        user_id = prf.user_config['datahub']['supermag']['username']

    extent = (dt_to - dt_fr).total_seconds() if extent is None else extent
    (success, sites_available) = smapi.SuperMAGGetInventory(user_id, dt_fr, extent)
    if not success:
        raise ConnectionError

    all_site_info = load_site_info()

    all_sites = all_site_info['CODE']
    need_update = []

    site_info_new = {
        'CODE': [],
        'NAME': [],
        'GLAT': [],
        'GLON': [],
        'OPERATORS': [],
    }
    for s in sites_available:
        ii = np.where(all_sites == s)[0]
        if not list(ii):
            need_update.append(s)
            continue
        site_info_new['CODE'].append(all_site_info['CODE'][ii[0]])
        site_info_new['NAME'].append(all_site_info['NAME'][ii[0]])
        site_info_new['GLAT'].append(all_site_info['GLAT'][ii[0]])
        site_info_new['GLON'].append(all_site_info['GLON'][ii[0]])
        site_info_new['OPERATORS'].append(all_site_info['OPERATORS'][ii[0]])

    if list(need_update):
        mylog.StreamLogger.warning("Cannot find the following sites, update the SuperMAG station records is needed!")
        mylog.simpleinfo.info(need_update)

    for k, v in site_info_new:
        site_info_new[k] = np.array(v)

    return site_info_new


def get_supermag_sites(
        dt_fr=None, dt_to=None, extent=None, user_id=None,
        lon_1=None, lon_2=None,
        lat_1=None, lat_2=None,
):
    lon_1 = 0. if lon_1 is None else lon_1
    lon_2 = 360. if lon_2 is None else lon_2

    lat_1 = - 90. if lat_1 is None else lat_1
    lat_2 = 90. if lat_2 is None else lat_2

    site_info = list_available_sites(dt_fr=dt_fr, dt_to=dt_to, extent=extent, user_id=user_id)

    glons = site_info['GLON']
    glats = site_info['GLAT']
    if lon_1 < lon_2:
        inds_lon = np.where((glons >= lon_1) & (glons < lon_2))[0]
    else:
        inds_lon = np.where((glons >= lon_1) & (glons < 360.))[0]
        inds_lon = np.concatenate((inds_lon, np.where((glons >=0) & (glons < lon_2))[0]))
    inds_lat = np.where((glats >= lat_1) & (glats <= lat_2))[0]
    inds = np.intersect1d(inds_lat, inds_lon)
    if not list(inds):
        return None

    site_info_new = {}
    for k, v in site_info.items():
        site_info_new[k] = v[inds]

    return site_info_new


def load_site_info(file_path=None, ):
    if file_path is None:
        file_path = pathlib.Path("SuperMAG_stations.dat")

    site_info = {
        'CODE': [],
        'NAME': [],
        'GLAT': [],
        'GLON': [],
        'OPERATORS': [],
    }
    with open(file_path, 'r') as f:
        text = f.read()

        results = re.findall(
            r'^(\w+)\s+([-.\d]+)\s+([-.\d]+)\s+([-.\d]+)\s+([-.\d]+)\s+"([^"]*)"\s+(\d+)\s+(.+)',
            text,
            re.M
        )
        results = list(zip(*results))
        site_info['CODE'] = np.array(results[0])
        site_info['NAME'] = np.array(results[5])
        site_info['GLAT'] = np.array(results[2]).astype(np.float32)
        site_info['GLON'] = np.array(results[1]).astype(np.float32) % 360

        ops = results[7]
        ops = [[i.strip('"') for i in op.split('\t')] for op in ops]
        site_info['OPERATORS'] = np.array(ops)

    return site_info


if __name__ == "__main__":
    # load_site_info()
    dt_fr_1 = datetime.datetime(2016, 3, 14)
    list_available_sites(dt_fr=dt_fr_1, extent=3600, user_id='JouleCai')