# Licensed under the BSD 3-Clause License
# Copyright (C) 2021 GeospaceLab (geospacelab)
# Author: Lei Cai, Space Physics and Astronomy, University of Oulu

__author__ = "Lei Cai"
__copyright__ = "Copyright 2021, GeospaceLab"
__license__ = "BSD-3-Clause License"
__email__ = "lei.cai@oulu.fi"
__docformat__ = "reStructureText"


import numpy as np
import datetime
import matplotlib.pyplot as plt
import geospacelab.cs as cs


def example_cs_for_user_own_data():
    # create the data arrays for the geographic coordinates (lat, lon, height)
    geo_lat = np.arange(-90., 90, 1.)
    geo_lon = np.ones_like(geo_lat) * 30.
    ut = datetime.datetime(2015, 3, 18, 12)

    # create a GEO object and assign coordinate arrays
    cs_geo = cs.set_cs('GEO')
    cs_geo['lat'] = geo_lat
    cs_geo['lon'] = geo_lon
    cs_geo['height'] = 120.     # in km
    cs_geo.ut = ut

    # transformation from GEO to AACGM
    cs_aacgm = cs_geo.to_AACGM(append_mlt=True)
    aacgm_lat = cs_aacgm['lat']

    # simple plot
    plt.plot(geo_lat, aacgm_lat)
    plt.xlabel('GLAT')
    plt.ylabel('MLAT')
    plt.show()
    return


if __name__ == "__main__":
    example_cs_for_user_own_data()
