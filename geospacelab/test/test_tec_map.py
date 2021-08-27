import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

import geospacelab.visualization.mpl.map_proj.geomap_viewer as geomap


def test_tec():
    dt_fr = datetime.datetime(2016, 3, 15, 12)
    dt_to = datetime.datetime(2016, 3, 16, 2)
    viewer = geomap.GeoMapViewer(dt_fr=dt_fr, dt_to=dt_to, figure_config={'figsize': (10, 5)})
    viewer.dock(datasource_contents=['madrigal', 'gnss_tec'])
    viewer.set_layout(1, 2)

    tec = viewer.assign_variable('TEC_MAP', dataset_index=1)
    dts = viewer.assign_variable('DATETIME', dataset_index=1).value.flatten()
    glat = viewer.assign_variable('GEO_LAT', dataset_index=1).value
    glon = viewer.assign_variable('GEO_LON', dataset_index=1).value

    time1 = datetime.datetime(2016, 3, 15, 14, 10)
    ind_t = np.where(dts == time1)[0]

    pid = viewer.add_polar_map(row_ind=0, col_ind=0, style='mlt-fixed', cs='AACGM', mlt_c=0., pole='N', ut=time1)
    # pid = viewer.add_polar_map(row_ind=0, col_ind=0, style='lst-fixed', cs='GEO', lst_c=0., pole='N', ut=time1)
    panel1 = viewer.panels[pid]

    tec_ = tec.value[ind_t[0], :, :]
    pcolormesh_config = tec.visual.plot_config.pcolormesh
    pcolormesh_config.update(c_lim=[0, 35])
    panel1.add_pcolor(tec_, coords={'lat': glat, 'lon': glon, 'height': 250.}, cs='GEO', **pcolormesh_config)
    panel1.add_coastlines()
    panel1.add_grids()

    plt.show()


if __name__ == "__main__":
    test_tec()
