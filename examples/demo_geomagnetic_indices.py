# Licensed under the BSD 3-Clause License
# Copyright (C) 2021 GeospaceLab (geospacelab)
# Author: Lei Cai, Space Physics and Astronomy, University of Oulu

__author__ = "Lei Cai"
__copyright__ = "Copyright 2021, GeospaceLab"
__license__ = "BSD-3-Clause License"
__email__ = "lei.cai@oulu.fi"
__docformat__ = "reStructureText"

import datetime
import matplotlib.pyplot as plt

import geospacelab.visualization.mpl.dashboards as dashboards


def quicklook_geomag_indices():
    dt_fr = datetime.datetime(2015, 2, 15, 0)
    dt_to = datetime.datetime(2015, 2, 16, 12)
    tsdb = dashboards.TSDashboard(dt_fr=dt_fr, dt_to=dt_to, figure_config={'figsize': (12, 8)})
    ds1 = tsdb.dock(datasource_contents=['wdc', 'asysym'])
    # ds2 = tsdb.dock(datasource_contents=['wdc', 'ae'])
    ds2 = tsdb.dock(datasource_contents=['supermag', 'indices'])
    ds3 = tsdb.dock(datasource_contents=['wdc', 'dst'])

    ds4 = tsdb.dock(datasource_contents=['gfz', 'kpap'])
    ds5 = tsdb.dock(datasource_contents=['gfz', 'hpo'], data_res=30)
    ds6 = tsdb.dock(datasource_contents=['gfz', 'snf107'])

    asy_d = ds1['ASY_D']
    asy_h = ds1['ASY_H']
    sym_d = ds1['SYM_D']
    sym_h = ds1['SYM_H']

    ae = ds2['SME']
    au = ds2['SMU']
    al = ds2['SML']

    dst = ds3['Dst']

    kp = ds4['Kp']
    hp = ds5['Hp']

    layout = [[dst], [asy_d, asy_h, sym_d, sym_h], [ae, au, al], [kp], [hp]]
    tsdb.set_layout(panel_layouts=layout, hspace=0.1)
    tsdb.draw()
    tsdb.show()
    pass


if __name__ == "__main__":
    quicklook_geomag_indices()
    # plt.savefig('tmp')
