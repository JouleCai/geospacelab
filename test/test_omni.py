import datetime
import matplotlib.pyplot as plt

import geospacelab.visualization.mpl.ts_viewer as ts


def test():
    dt_fr = datetime.datetime(2016, 3, 13, 12)
    dt_to = datetime.datetime(2016, 3, 16, 12)
    tsviewer = ts.TimeSeriesViewer(dt_fr=dt_fr, dt_to=dt_to)
    ds1 = tsviewer.dock(datasource_contents=['cdaweb', 'omni'])
    ds1.load_data()

    Bx = tsviewer.assign_variable('B_x_GSM')
    By = tsviewer.assign_variable('B_y_GSM')
    Bz = tsviewer.assign_variable('B_z_GSM')

    n_p = tsviewer.assign_variable('n_p')
    v_sw = tsviewer.assign_variable('v_sw')
    p_dyn = tsviewer.assign_variable('p_dyn')

    layout = [[Bx, By, Bz], [v_sw], [n_p], [p_dyn]]
    tsviewer.set_layout(panel_layouts=layout, hspace=0.1)
    tsviewer.draw()
    pass


if __name__ == "__main__":
    test()
    plt.show()
