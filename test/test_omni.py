import datetime
import matplotlib.pyplot as plt

import geospacelab.visualization.mpl.ts_viewer as ts


def test():
    dt_fr = datetime.datetime(2014, 11, 29, 0)
    dt_to = datetime.datetime(2014, 12, 1, 23)
    tsviewer = ts.TimeSeriesViewer(dt_fr=dt_fr, dt_to=dt_to)
    ds1 = tsviewer.dock(datasource_contents=['cdaweb', 'omni'])
    ds1.load_data()

    Bx = tsviewer.assign_variable('B_x_GSM')
    By = tsviewer.assign_variable('B_y_GSM')
    Bz = tsviewer.assign_variable('B_z_GSM')

    n_p = tsviewer.assign_variable('n_p')
    v_sw = tsviewer.assign_variable('v_sw')
    p_dyn = tsviewer.assign_variable('p_dyn')

    ds2 = tsviewer.dock(datasource_contents=['wdc', 'asysym'])

    sym_h = tsviewer.assign_variable('SYM_H')

    ds3 = tsviewer.dock(datasource_contents=['wdc', 'ae'])

    ae = tsviewer.assign_variable('AE')
    au = tsviewer.assign_variable('AU')
    al = tsviewer.assign_variable('AL')

    layout = [[Bx, By, Bz], [v_sw], [n_p], [p_dyn], [sym_h], [ae, au, al]]

    tsviewer.set_layout(panel_layouts=layout, hspace=0.1)
    tsviewer.draw()
    tsviewer.save_figure(file_name='example_omni_6', append_time=False)
    pass


if __name__ == "__main__":
    test()
    plt.show()
