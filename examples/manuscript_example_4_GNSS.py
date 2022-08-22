import datetime
import numpy as np
import matplotlib.pyplot as plt
from geospacelab import preferences as pref
# pref.user_config['visualization']['mpl']['style'] = 'dark'

import geospacelab.visualization.mpl.geomap.geodashboards as geomap


def test_tec():

    dt_fr = datetime.datetime(2015, 3, 17, 1)
    dt_to = datetime.datetime(2015, 3, 17, 23)
    viewer = geomap.GeoDashboard(dt_fr=dt_fr, dt_to=dt_to, figure_config={'figsize': (15, 5)})
    viewer.dock(datasource_contents=['madrigal', 'gnss', 'tecmap'])
    viewer.set_layout(1, 3, left=0.05, right=0.9, wspace=0.5)

    tec = viewer.assign_variable('TEC_MAP', dataset_index=1)
    dts = viewer.assign_variable('DATETIME', dataset_index=1).value.flatten()
    glat = viewer.assign_variable('GEO_LAT', dataset_index=1).value
    glon = viewer.assign_variable('GEO_LON', dataset_index=1).value

    time_c = datetime.datetime(2015, 3, 17, 6, 25)
    ind_t = np.where(dts == time_c)[0]

    # panel1 = viewer.add_polar_map(row_ind=0, col_ind=0, style='mlt-fixed', cs='AACGM', mlt_c=0., pole='N', ut=time_c, boundary_lat=60)
    # pid = viewer.add_polar_map(row_ind=0, col_ind=0, style='mlt-fixed', cs='AACGM', mlt_c=0., pole='S', ut=time_c, mirror_south=True)
    # pid = viewer.add_polar_map(row_ind=0, col_ind=0, style='lst-fixed', cs='GEO', lst_c=0., pole='N', ut=time_c, boundary_lat=60)
    # pid = viewer.add_polar_map(row_ind=0, col_ind=0, style='lst-fixed', cs='GEO', lst_c=0, pole='S', ut=time_c, mirror_south=True)
    panel1 = viewer.add_polar_map(row_ind=0, col_ind=0, style='lon-fixed', cs='GEO', lon_c=0., pole='N', ut=time_c,
                                  boundary_lat=30., mirror_south=False)
    # pid = viewer.add_polar_map(row_ind=0, col_ind=0, style='lon-fixed', cs='GEO', lon_c=0., pole='N', ut=time_c,
    #                        boundary_lat=30, mirror_south=False)
    panel1.overlay_coastlines()
    panel1.overlay_gridlines()
    #
    tec_ = tec.value[ind_t[0], :, :]
    pcolormesh_config = tec.visual.plot_config.pcolormesh
    pcolormesh_config.update(c_lim=[5, 30])

    import geospacelab.visualization.mpl.colormaps as cm
    pcolormesh_config.update(cmap='jet')
    ipc = panel1.overlay_pcolormesh(tec_, coords={'lat': glat, 'lon': glon, 'height': 250.}, cs='GEO', **pcolormesh_config)
    panel1.add_colorbar(ipc, c_label="TECU", c_scale='linear', left=1.1, bottom=0.1, width=0.05, height=0.7)
    #
    # # viewer.add_text(0.5, 1.1, "dashboard title")
    panel1.add_title(title='Geographic longitude-fixed')

    ##############################################################3
    time_c = datetime.datetime(2015, 3, 17, 6, 30)
    ind_t = np.where(dts == time_c)[0]

    # panel1 = viewer.add_polar_map(row_ind=0, col_ind=0, style='mlt-fixed', cs='AACGM', mlt_c=0., pole='N', ut=time_c, boundary_lat=60)
    # pid = viewer.add_polar_map(row_ind=0, col_ind=0, style='mlt-fixed', cs='AACGM', mlt_c=0., pole='S', ut=time_c, mirror_south=True)
    panel1 = viewer.add_polar_map(row_ind=0, col_ind=1, style='lst-fixed', cs='GEO', lst_c=0., pole='N', ut=time_c, boundary_lat=30)
    # pid = viewer.add_polar_map(row_ind=0, col_ind=0, style='lst-fixed', cs='GEO', lst_c=0, pole='S', ut=time_c, mirror_south=True)
    # panel1 = viewer.add_polar_map(row_ind=0, col_ind=0, style='lon-fixed', cs='GEO', lon_c=0., pole='N', ut=time_c,
    #                               boundary_lat=40., mirror_south=False)
    # pid = viewer.add_polar_map(row_ind=0, col_ind=0, style='lon-fixed', cs='GEO', lon_c=0., pole='N', ut=time_c,
    #                        boundary_lat=30, mirror_south=False)
    panel1.overlay_coastlines()
    panel1.overlay_gridlines()
    #
    tec_ = tec.value[ind_t[0], :, :]
    pcolormesh_config = tec.visual.plot_config.pcolormesh
    pcolormesh_config.update(c_lim=[5, 30])

    import geospacelab.visualization.mpl.colormaps as cm
    pcolormesh_config.update(cmap='jet')
    ipc = panel1.overlay_pcolormesh(tec_, coords={'lat': glat, 'lon': glon, 'height': 250.}, cs='GEO',
                                    **pcolormesh_config)
    panel1.add_colorbar(ipc, c_label="TECU", c_scale='linear', left=1.1, bottom=0.1, width=0.05, height=0.7)
    #
    # # viewer.add_text(0.5, 1.1, "dashboard title")
    panel1.add_title(title='Geographic LST-fixed', transform=panel1().transAxes)

    ##############################################################3
    time_c = datetime.datetime(2015, 3, 17, 6, 30)
    ind_t = np.where(dts == time_c)[0]

    panel1 = viewer.add_polar_map(row_ind=0, col_ind=2, style='mlt-fixed', cs='APEX', mlt_c=0., pole='N', ut=time_c, boundary_lat=30)
    # pid = viewer.add_polar_map(row_ind=0, col_ind=0, style='mlt-fixed', cs='AACGM', mlt_c=0., pole='S', ut=time_c, mirror_south=True)
    # panel1 = viewer.add_polar_map(row_ind=0, col_ind=1, style='lst-fixed', cs='GEO', lst_c=0., pole='N', ut=time_c, boundary_lat=40)
    # pid = viewer.add_polar_map(row_ind=0, col_ind=0, style='lst-fixed', cs='GEO', lst_c=0, pole='S', ut=time_c, mirror_south=True)
    # panel1 = viewer.add_polar_map(row_ind=0, col_ind=0, style='lon-fixed', cs='GEO', lon_c=0., pole='N', ut=time_c,
    #                               boundary_lat=40., mirror_south=False)
    # pid = viewer.add_polar_map(row_ind=0, col_ind=0, style='lon-fixed', cs='GEO', lon_c=0., pole='N', ut=time_c,
    #                        boundary_lat=30, mirror_south=False)
    panel1.overlay_coastlines()
    panel1.overlay_gridlines()
    #
    tec_ = tec.value[ind_t[0], :, :]
    pcolormesh_config = tec.visual.plot_config.pcolormesh
    pcolormesh_config.update(c_lim=[5, 30])

    import geospacelab.visualization.mpl.colormaps as cm
    pcolormesh_config.update(cmap='jet')
    ipc = panel1.overlay_pcolormesh(tec_, coords={'lat': glat, 'lon': glon, 'height': 250.}, cs='GEO',
                                    **pcolormesh_config)
    panel1.add_colorbar(ipc, c_label="TECU", c_scale='linear', left=1.1, bottom=0.1, width=0.05, height=0.7)
    #
    # # viewer.add_text(0.5, 1.1, "dashboard title")
    panel1.add_title(title='APEX MLT-fixed')

    plt.savefig('example_tec_aacgm_fixed_mlt', dpi=200)
    plt.show()


if __name__ == "__main__":
    test_tec()
