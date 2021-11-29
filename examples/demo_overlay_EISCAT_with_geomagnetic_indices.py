import datetime
import matplotlib.pyplot as plt

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

    layout = [[n_e, [al]], [T_e], [T_i], [v_i], [kp]]
    dashboard.set_layout(panel_layouts=layout, )
    dashboard.draw()
    dashboard.add_title()
    dashboard.add_panel_labels()
    plt.show()


if __name__ == '__main__':
    test_combine()
