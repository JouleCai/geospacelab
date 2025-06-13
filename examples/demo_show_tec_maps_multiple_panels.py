# Licensed under the BSD 3-Clause License
# Copyright (C) 2021 GeospaceLab (geospacelab)
# Author: Lei Cai, Space Physics and Astronomy, University of Oulu

__author__ = "Lei Cai"
__copyright__ = "Copyright 2021, GeospaceLab"
__license__ = "BSD-3-Clause License"
__email__ = "lei.cai@oulu.fi"
__docformat__ = "reStructureText"


import datetime
import numpy as np
import matplotlib.pyplot as plt

# pref.user_config['visualization']['mpl']['style'] = 'dark'

import geospacelab.visualization.mpl.geomap.geodashboards as geomap


def test_tec():

    dt_fr = datetime.datetime(2021, 8, 24, 1)
    dt_to = datetime.datetime(2021, 8, 24, 23)
    db = geomap.GeoDashboard(dt_fr=dt_fr, dt_to=dt_to, figure_config={'figsize': (15, 10)})
    ds_tec = db.dock(datasource_contents=['madrigal', 'gnss', 'tecmap'])
    db.set_layout(2, 3, wspace=0.5)

    tec = ds_tec['TEC_MAP']
    dts = ds_tec['DATETIME'].flatten()
    glat = ds_tec['GEO_LAT'].value
    glon = ds_tec['GEO_LON'].value

    """
    Generation of the first panel
    """
    time_c = datetime.datetime(2021, 8, 24, 9, 30)
    ind_t = np.where(dts == time_c)[0]

    # Add the first panel
    # AACGM LAT-MLT in the northern hemisphere
    panel = db.add_polar_map(row_ind=0, col_ind=0, style='mlt-fixed', cs='AACGM', mlt_c=0., pole='N', ut=time_c, boundary_lat=50)
    # AACGM LAT-MLT in the southern hemisphere
    # panel = db.add_polar_map(row_ind=0, col_ind=0, style='mlt-fixed', cs='AACGM', mlt_c=0., pole='S', ut=time_c, mirror_south=True)
    # GEO LAT-LST in the northern hemisphere
    # panel = db.add_polar_map(row_ind=0, col_ind=0, style='lst-fixed', cs='GEO', lst_c=0., pole='N', ut=time_c, boundary_lat=50)
    # GEO LAT-LST in the southern hemisphere
    # panel = db.add_polar_map(row_ind=0, col_ind=0, style='lst-fixed', cs='GEO', lst_c=0, pole='S', ut=time_c, mirror_south=True)
    # GEO LAT-LON in the southern hemisphere
    # panel = db.add_polar_map(row_ind=0, col_ind=0, style='lon-fixed', cs='GEO', lon_c=0., pole='S', ut=time_c,
    #                          boundary_lat=0, mirror_south=False)
    # GEO LAT-LON in the northern hemisphere
    # pid = db.add_polar_map(row_ind=0, col_ind=0, style='lon-fixed', cs='GEO', lon_c=0., pole='N', ut=time_c,
    #                        boundary_lat=30, mirror_south=False)
    panel.overlay_coastlines()
    panel.overlay_gridlines()
    #
    # retrieve the data array
    tec_ = tec.value[ind_t[0]]
    # Configuration for plotting
    pcolormesh_config = tec.visual.plot_config.pcolormesh
    pcolormesh_config.update(c_lim=[0, 20])
    import geospacelab.visualization.mpl.colormaps as cm
    pcolormesh_config.update(cmap='turbo')

    # overlay the 2-D TEC map
    ipc = panel.overlay_pcolormesh(tec_, coords={'lat': glat, 'lon': glon, 'height': 250.}, cs='GEO', **pcolormesh_config)
    panel.add_colorbar(ipc, c_label="TECU", c_scale='linear', left=1.1, bottom=0.1, width=0.05, height=0.7)
    # add the panel title
    panel.add_title(title=time_c.strftime("%Y-%m-%d %H:%M"))

    """
    Repeating process for the second panel
    """
    time_c = datetime.datetime(2021, 8, 24, 10, 0)
    ind_t = np.where(dts == time_c)[0]

    panel = db.add_polar_map(row_ind=0, col_ind=1, style='mlt-fixed', cs='AACGM', mlt_c=0., pole='N', ut=time_c, boundary_lat=50)

    panel.overlay_coastlines()
    panel.overlay_gridlines()
    #
    tec_ = tec.value[ind_t[0], :, :]
    pcolormesh_config = tec.visual.plot_config.pcolormesh
    pcolormesh_config.update(c_lim=[0, 20])

    import geospacelab.visualization.mpl.colormaps as cm
    pcolormesh_config.update(cmap='turbo')
    ipc = panel.overlay_pcolormesh(tec_, coords={'lat': glat, 'lon': glon, 'height': 250.}, cs='GEO', **pcolormesh_config)
    panel.add_colorbar(ipc, c_label="TECU", c_scale='linear', left=1.1, bottom=0.1, width=0.05, height=0.7)
    panel.add_title(title=time_c.strftime("%Y-%m-%d %H:%M"))

    """
    Repeating process for the third panel
    """
    time_c = datetime.datetime(2021, 8, 24, 10, 30)
    ind_t = np.where(dts == time_c)[0]

    panel = db.add_polar_map(row_ind=0, col_ind=2, style='mlt-fixed', cs='AACGM', mlt_c=0., pole='N', ut=time_c, boundary_lat=50)

    panel.overlay_coastlines()
    panel.overlay_gridlines()
    #
    tec_ = tec.value[ind_t[0], :, :]
    pcolormesh_config = tec.visual.plot_config.pcolormesh
    pcolormesh_config.update(c_lim=[0, 20])

    import geospacelab.visualization.mpl.colormaps as cm
    pcolormesh_config.update(cmap='turbo')
    ipc = panel.overlay_pcolormesh(tec_, coords={'lat': glat, 'lon': glon, 'height': 250.}, cs='GEO', **pcolormesh_config)
    panel.add_colorbar(ipc, c_label="TECU", c_scale='linear', left=1.1, bottom=0.1, width=0.05, height=0.7)
    panel.add_title(title=time_c.strftime("%Y-%m-%d %H:%M"))

    """
    Repeating process for the fourth panel
    """
    time_c = datetime.datetime(2021, 8, 24, 11, 0)
    ind_t = np.where(dts == time_c)[0]

    panel = db.add_polar_map(row_ind=1, col_ind=0, style='mlt-fixed', cs='AACGM', mlt_c=0., pole='N', ut=time_c, boundary_lat=50)

    panel.overlay_coastlines()
    panel.overlay_gridlines()
    #
    tec_ = tec.value[ind_t[0], :, :]
    pcolormesh_config = tec.visual.plot_config.pcolormesh
    pcolormesh_config.update(c_lim=[0, 20])

    import geospacelab.visualization.mpl.colormaps as cm
    pcolormesh_config.update(cmap='turbo')
    ipc = panel.overlay_pcolormesh(tec_, coords={'lat': glat, 'lon': glon, 'height': 250.}, cs='GEO', **pcolormesh_config)
    panel.add_colorbar(ipc, c_label="TECU", c_scale='linear', left=1.1, bottom=0.1, width=0.05, height=0.7)
    panel.add_title(title=time_c.strftime("%Y-%m-%d %H:%M"))

    """
    Repeating process for the fifth panel
    """
    time_c = datetime.datetime(2021, 8, 24, 11, 30)
    ind_t = np.where(dts == time_c)[0]

    panel = db.add_polar_map(row_ind=1, col_ind=1, style='mlt-fixed', cs='AACGM', mlt_c=0., pole='N', ut=time_c, boundary_lat=50)
    panel.overlay_coastlines()
    panel.overlay_gridlines()
    #
    tec_ = tec.value[ind_t[0], :, :]
    pcolormesh_config = tec.visual.plot_config.pcolormesh
    pcolormesh_config.update(c_lim=[0, 20])

    import geospacelab.visualization.mpl.colormaps as cm
    pcolormesh_config.update(cmap='turbo')
    ipc = panel.overlay_pcolormesh(tec_, coords={'lat': glat, 'lon': glon, 'height': 250.}, cs='GEO', **pcolormesh_config)
    panel.add_colorbar(ipc, c_label="TECU", c_scale='linear', left=1.1, bottom=0.1, width=0.05, height=0.7)
    panel.add_title(title=time_c.strftime("%Y-%m-%d %H:%M"))

    """
        Repeating process for the sixth panel
    """
    time_c = datetime.datetime(2021, 8, 24, 12, 0)
    ind_t = np.where(dts == time_c)[0]

    panel = db.add_polar_map(row_ind=1, col_ind=2, style='mlt-fixed', cs='AACGM', mlt_c=0., pole='N', ut=time_c, boundary_lat=50)

    panel.overlay_coastlines()
    panel.overlay_gridlines()
    #
    tec_ = tec.value[ind_t[0], :, :]
    pcolormesh_config = tec.visual.plot_config.pcolormesh
    pcolormesh_config.update(c_lim=[0, 20])

    import geospacelab.visualization.mpl.colormaps as cm
    pcolormesh_config.update(cmap='turbo')
    ipc = panel.overlay_pcolormesh(tec_, coords={'lat': glat, 'lon': glon, 'height': 250.}, cs='GEO', **pcolormesh_config)
    panel.add_colorbar(ipc, c_label="TECU", c_scale='linear', left=1.1, bottom=0.1, width=0.05, height=0.7)
    panel.add_title(title=time_c.strftime("%Y-%m-%d %H:%M"))

    plt.savefig('example_tec_aacgm_fixed_mlt', dpi=300)
    plt.show()


if __name__ == "__main__":
    test_tec()
