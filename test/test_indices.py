import datetime
import matplotlib.pyplot as plt

import geospacelab.visualization.mpl.dashboards as dashboards


def test():
    dt_fr = datetime.datetime(2005, 8, 31, 12)
    dt_to = datetime.datetime(2005, 9, 2, 12)
    tsdb = dashboards.TSDashboard(dt_fr=dt_fr, dt_to=dt_to, figure_config={'figsize': (15, 8)})
    ds1 = tsdb.dock(datasource_contents=['wdc', 'asysym'])
    ds2 = tsdb.dock(datasource_contents=['wdc', 'ae'])
    ds3 = tsdb.dock(datasource_contents=['wdc', 'dst'])
    
    ds4 = tsdb.dock(datasource_contents=['gfz', 'kpap'])
    ds5 = tsdb.dock(datasource_contents=['gfz', 'hpo'], data_res=60)
    # ds1.load_data()

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
    plt.savefig('example_geomagnetic_indices.png')
    pass


if __name__ == "__main__":
    test()
    plt.show()
