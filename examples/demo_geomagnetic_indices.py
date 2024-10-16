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
    ds2 = tsdb.dock(datasource_contents=['wdc', 'ae'])
    ds3 = tsdb.dock(datasource_contents=['wdc', 'dst'])

    ds4 = tsdb.dock(datasource_contents=['gfz', 'kpap'])
    ds5 = tsdb.dock(datasource_contents=['gfz', 'hpo'], data_res=60)
    ds6 = tsdb.dock(datasource_contents=['gfz', 'snf107'])

    asy_d = tsdb.assign_variable('ASY_D', dataset=ds1)
    asy_h = tsdb.assign_variable('ASY_H', dataset=ds1)
    sym_d = tsdb.assign_variable('SYM_D', dataset=ds1)
    sym_h = tsdb.assign_variable('SYM_H', dataset=ds1)

    ae = tsdb.assign_variable('AE', dataset=ds2)
    au = tsdb.assign_variable('AU', dataset=ds2)
    al = tsdb.assign_variable('AL', dataset=ds2)
    ao = tsdb.assign_variable('AO', dataset=ds2)

    dst = tsdb.assign_variable('Dst', dataset=ds3)

    kp = tsdb.assign_variable('Kp', dataset=ds4)
    hp = tsdb.assign_variable('Hp', dataset=ds5)

    layout = [[dst], [asy_d, asy_h, sym_d, sym_h], [ae, au, ao, al], [kp], [hp]]
    tsdb.set_layout(panel_layouts=layout, hspace=0.1)
    tsdb.draw()
    pass


if __name__ == "__main__":
    quicklook_geomag_indices()
    plt.savefig('tmp')
