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
import geospacelab.express.eiscat_dashboard as eiscat


def test_combine():
    dt_fr = datetime.datetime.strptime('20050910' + '1200', '%Y%m%d%H%M')
    dt_to = datetime.datetime.strptime('20050913' + '1200', '%Y%m%d%H%M')

    site = 'UHF'
    antenna = 'UHF'
    modulation = ''
    load_mode = 'AUTO'
    data_file_type = 'eiscat-hdf5'

    dashboard = eiscat.EISCATDashboard(dt_fr, dt_to, site=site, antenna=antenna, modulation=modulation,
                                 data_file_type=data_file_type, load_mode=load_mode, status_control=True,
                                 residual_control=True)

    # select beams before assign the variables
    # dashboard.dataset.select_beams(field_aligned=True)
    # dashboard.dataset.select_beams(az_el_pairs=[(188.6, 77.7)])
    dashboard.check_beams()

    dashboard.status_mask()

    n_e = dashboard.assign_variable('n_e')
    T_i = dashboard.assign_variable('T_i')
    T_e = dashboard.assign_variable('T_e')
    v_i = dashboard.assign_variable('v_i_los')
    az = dashboard.assign_variable('AZ')
    el = dashboard.assign_variable('EL')
    ptx = dashboard.assign_variable('P_Tx')
    tsys = dashboard.assign_variable('T_SYS_1')

    ds2 = dashboard.dock(datasource_contents=['gfz', 'kpap'])
    kp = dashboard.assign_variable('Kp', dataset=ds2)
    ds3 = dashboard.dock(datasource_contents=['wdc', 'ae'])
    ae = dashboard.assign_variable('AE', dataset=ds3)
    al = dashboard.assign_variable('AL', dataset=ds3)


    layout = [[n_e, [al]], [T_i], [v_i], [kp]]
    dashboard.set_layout(panel_layouts=layout, top=0.5, bottom=0.05)
    dashboard.draw()
    dashboard.add_title()
    dashboard.add_panel_labels()

############################33############################3
    # add another dashboard
    dt_fr = datetime.datetime(2005, 9, 10, 0)
    dt_to = datetime.datetime(2005, 10, 14, 0)
    dashboard = dashboards.TSDashboard(dt_fr=dt_fr, dt_to=dt_to)
    ds1 = dashboard.dock(datasource_contents=['cdaweb', 'omni'])
    ds1.load_data()

    Bx = dashboard.assign_variable('B_x_GSM')
    By = dashboard.assign_variable('B_y_GSM')
    Bz = dashboard.assign_variable('B_z_GSM')

    n_p = dashboard.assign_variable('n_p')
    v_sw = dashboard.assign_variable('v_sw')
    p_dyn = dashboard.assign_variable('p_dyn')

    ds2 = dashboard.dock(datasource_contents=['wdc', 'asysym'])

    sym_h = dashboard.assign_variable('SYM_H')

    ds3 = dashboard.dock(datasource_contents=['wdc', 'ae'])

    ae = dashboard.assign_variable('AE')
    au = dashboard.assign_variable('AU')
    al = dashboard.assign_variable('AL')

    layout = [[Bx, By, Bz], [v_sw], [p_dyn], [sym_h]]
    # layout = [[Bz, By], [v_sw], [n_p], [sym_h]]
    dashboard.set_layout(panel_layouts=layout, hspace=0.1, top=0.95, bottom=0.65, left=0.1, right=0.9)
    dashboard.draw()

    plt.show()


if __name__ == '__main__':
    test_combine()