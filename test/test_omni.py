import datetime
import matplotlib.pyplot as plt

import geospacelab.visualization.mpl.dashboards as dashboards


def test():
    dt_fr = datetime.datetime(2021, 3, 23, 0)
    dt_to = datetime.datetime(2021, 3, 23, 23)
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

    ds4 = dashboard.dock(datasource_contents=['gfz', 'kpap'])
    kp = dashboard.assign_variable('Kp')

    layout = [[Bx, By, Bz], [v_sw], [n_p], [p_dyn], [sym_h], [ae, au, al], [kp]]
    # layout = [[Bz, By], [v_sw], [n_p], [sym_h]]
    dashboard.set_layout(panel_layouts=layout, hspace=0.1)
    dashboard.draw()
    # dashboard.save_figure(file_name='example_omni_6', append_time=False)
    pass


if __name__ == "__main__":
    test()
    plt.show()
