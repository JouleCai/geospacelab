import datetime
import numpy as np
import matplotlib.pyplot as plt

import geospacelab.visualization.map_proj.geomap_viewer as geomap


def test_tec():
<<<<<<< HEAD
    dt_fr = datetime.datetime(2015, 2, 15, 0)
    dt_to = datetime.datetime(2015, 2, 15, 23)
=======
    dt_fr = datetime.datetime(2021, 8, 24, 0)
    dt_to = datetime.datetime(2021, 8, 24, 12)
>>>>>>> master
    viewer = geomap.GeoMapViewer(dt_fr=dt_fr, dt_to=dt_to, figure_config={'figsize': (5, 5)})
    viewer.dock(datasource_contents=['madrigal', 'gnss', 'tecmap'])
    viewer.set_layout(1, 1)

    tec = viewer.assign_variable('TEC_MAP', dataset_index=1)
    dts = viewer.assign_variable('DATETIME', dataset_index=1).value.flatten()
    glat = viewer.assign_variable('GEO_LAT', dataset_index=1).value
    glon = viewer.assign_variable('GEO_LON', dataset_index=1).value

<<<<<<< HEAD
    time1 = datetime.datetime(2015, 2, 15, 19, 0)
    ind_t = np.where(dts == time1)[0]

    pid = viewer.add_polar_map(row_ind=0, col_ind=0, style='mlt-fixed', cs='AACGM', mlt_c=0., pole='N', ut=time1, boundary_lat=50)
=======
    time1 = datetime.datetime(2021, 8, 24, 11, 10)
    ind_t = np.where(dts == time1)[0]

    # pid = viewer.add_polar_map(row_ind=0, col_ind=0, style='mlt-fixed', cs='AACGM', mlt_c=0., pole='N', ut=time1, boundary_lat=60)
>>>>>>> master
    # pid = viewer.add_polar_map(row_ind=0, col_ind=0, style='mlt-fixed', cs='AACGM', mlt_c=0., pole='S', ut=time1, mirror_south=True)
    pid = viewer.add_polar_map(row_ind=0, col_ind=0, style='lst-fixed', cs='GEO', lst_c=0., pole='N', ut=time1, boundary_lat=60)
    # pid = viewer.add_polar_map(row_ind=0, col_ind=0, style='lst-fixed', cs='GEO', lst_c=0, pole='S', ut=time1, mirror_south=True)
    # pid = viewer.add_polar_map(row_ind=0, col_ind=0, style='lon-fixed', cs='GEO', lon_c=0., pole='S', ut=time1,
    #                          boundary_lat=0, mirror_south=False)
    # pid = viewer.add_polar_map(row_ind=0, col_ind=0, style='lon-fixed', cs='GEO', lon_c=0., pole='N', ut=time1,
    #                        boundary_lat=30, mirror_south=False)
    panel1 = viewer.panels[pid]
    panel1.add_coastlines()
    panel1.add_gridlines()
    #
    tec_ = tec.value[ind_t[0], :, :]
    pcolormesh_config = tec.visual.plot_config.pcolormesh
<<<<<<< HEAD
    pcolormesh_config.update(c_lim=[0, 15])
=======
    pcolormesh_config.update(c_lim=[5, 15])
>>>>>>> master
    import geospacelab.visualization.mpl.colormaps as cm
    pcolormesh_config.update(cmap='jet')
    ipc = panel1.add_pcolor(tec_, coords={'lat': glat, 'lon': glon, 'height': 250.}, cs='GEO', **pcolormesh_config)
    panel1.add_colorbar(
       ipc, ax=panel1.major_ax, c_label="TECU", c_scale='linear',
       left=1.1, bottom=0.1, width=0.05, height=0.7
    )
    #
    # # viewer.add_text(0.5, 1.1, "dashboard title")
    # # panel1.major_ax.text(0.5, 1.0, 'panel title', transform=panel1.major_ax.transAxes)

    plt.savefig('example_tec_aacgm_fixed_mlt', dpi=200)
    plt.show()


if __name__ == "__main__":
    test_tec()
