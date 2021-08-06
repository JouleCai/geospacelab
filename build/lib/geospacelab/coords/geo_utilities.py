import numpy as np

def convert_local_az_el_range_to_geo(lat_0, lon_0, height_0, az, el, beam_range=None, radians = False):
    if radians:
        rd = 1.
    else:
        rd = np.pi / 180.
    lat_0 = lat_0 * rd
    lon_0 = lon_0 * rd
    az = az * rd
    el = el * rd


