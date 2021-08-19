import datetime
import matplotlib.pyplot as plt

import geospacelab.visualization.ts_viewer as ts


def test():
    dt_fr = datetime.datetime(2005, 8, 31, 12)
    dt_to = datetime.datetime(2005, 9, 2, 12)
    tsviewer = ts.TimeSeriesViewer(dt_fr=dt_fr, dt_to=dt_to, figure_config={'figsize': (15, 8)})
    ds1 = tsviewer.dock(datasource_contents=['wdc', 'asysym'])
    ds2 = tsviewer.dock(datasource_contents=['wdc', 'ae'])
    ds3 = tsviewer.dock(datasource_contents=['wdc', 'dst'])
    # ds1.load_data()

    asy_d = tsviewer.assign_variable('ASY_D', dataset=ds1)
    asy_h = tsviewer.assign_variable('ASY_H', dataset=ds1)
    sym_d = tsviewer.assign_variable('SYM_D', dataset=ds1)
    sym_h = tsviewer.assign_variable('SYM_H', dataset=ds1)

    ae = tsviewer.assign_variable('AE', dataset=ds2)
    au = tsviewer.assign_variable('AU', dataset=ds2)
    al = tsviewer.assign_variable('AL', dataset=ds2)
    ao = tsviewer.assign_variable('AO', dataset=ds2)

    dst = tsviewer.assign_variable('Dst', dataset=ds3)

    layout = [[dst], [asy_d, asy_h, sym_d, sym_h], [ae, au, ao, al]]
    tsviewer.set_layout(panel_layouts=layout, hspace=0.1)
    tsviewer.draw()
    pass


if __name__ == "__main__":
    test()
    plt.show()
