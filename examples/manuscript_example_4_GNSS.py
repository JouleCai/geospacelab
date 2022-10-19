import datetime
import numpy as np
import matplotlib.pyplot as plt
from geospacelab import preferences as pref
# pref.user_config['visualization']['mpl']['style'] = 'dark'

import geospacelab.visualization.mpl.geomap.geodashboards as geomap


def test_tec():

    dt_fr = datetime.datetime(2015, 3, 17, 1)
    dt_to = datetime.datetime(2015, 3, 17, 23)
    db = geomap.GeoDashboard(dt_fr=dt_fr, dt_to=dt_to, figure_config={'figsize': (15, 5)})
    db.dock(datasource_contents=['madrigal', 'gnss', 'tecmap'])
    db.set_layout(1, 3, left=0.05, right=0.9, wspace=0.5)

    tec = db.assign_variable('TEC_MAP', dataset_index=0)
    dts = db.assign_variable('DATETIME', dataset_index=0).value.flatten()
    glat = db.assign_variable('GEO_LAT', dataset_index=0).value
    glon = db.assign_variable('GEO_LON', dataset_index=0).value

    """
    Add the first panel showing the TEC map in GEO LAT-LON coordinates
    """
    # check the index for the selected time
    time_c = datetime.datetime(2015, 3, 17, 6, 25)
    ind_t = np.where(dts == time_c)[0]
    # Add the polar map panel
    panel = db.add_polar_map(row_ind=0, col_ind=0, style='lon-fixed', cs='GEO', lon_c=0., pole='N', ut=time_c,
                            boundary_lat=30., mirror_south=False)
    # Overlay with coastlines and gridlines
    panel.overlay_coastlines()
    panel.overlay_gridlines()
    # Retrieve the TEC data array
    tec_ = tec.value[ind_t[0]]
    # Optional settings for plotting
    pcolormesh_config = tec.visual.plot_config.pcolormesh
    pcolormesh_config.update(c_lim=[5, 30])
    pcolormesh_config.update(cmap='jet')
    # Overlay with the TEC map
    ipc = panel.overlay_pcolormesh(tec_, coords={'lat': glat, 'lon': glon, 'height': 250.}, cs='GEO', **pcolormesh_config)
    panel.add_colorbar(ipc, c_label="TECU", c_scale='linear', left=1.1, bottom=0.1, width=0.05, height=0.7)

    panel.add_title(title='Geographic longitude-fixed')

    """
    Repeating process for the second panel showing the TEC map in GEO LAT-LST coordinates
    """
    # check the index for the selected time
    time_c = datetime.datetime(2015, 3, 17, 6, 30)
    ind_t = np.where(dts == time_c)[0]
    # Add the polar map panel
    panel = db.add_polar_map(row_ind=0, col_ind=1, style='lst-fixed', cs='GEO', lst_c=0., pole='N', ut=time_c, boundary_lat=30)
    # Overlay with coastlines and gridlines
    panel.overlay_coastlines()
    panel.overlay_gridlines()
    # Retrieve the TEC data array
    tec_ = tec.value[ind_t[0]]
    # Optional settings for plotting
    pcolormesh_config = tec.visual.plot_config.pcolormesh
    pcolormesh_config.update(c_lim=[5, 30])
    pcolormesh_config.update(cmap='jet')
    # Overlay with the TEC map
    ipc = panel.overlay_pcolormesh(tec_, coords={'lat': glat, 'lon': glon, 'height': 250.}, cs='GEO',
                                    **pcolormesh_config)
    panel.add_colorbar(ipc, c_label="TECU", c_scale='linear', left=1.1, bottom=0.1, width=0.05, height=0.7)

    panel.add_title(title='Geographic LST-fixed', transform=panel().transAxes)

    """
    Repeating process for the second panel showing the TEC map in APEX LAT-MLT coordinates
    """
    # check the index for the selected time
    time_c = datetime.datetime(2015, 3, 17, 6, 30)
    ind_t = np.where(dts == time_c)[0]
    # Add the polar map panel
    panel = db.add_polar_map(row_ind=0, col_ind=2, style='mlt-fixed', cs='APEX', mlt_c=0., pole='N', ut=time_c, boundary_lat=30)
    # Overlay with coastlines and gridlines
    panel.overlay_coastlines()
    panel.overlay_gridlines()
    # Retrieve the TEC data array
    tec_ = tec.value[ind_t[0]]
    # Optional settings for plotting
    pcolormesh_config = tec.visual.plot_config.pcolormesh
    pcolormesh_config.update(c_lim=[5, 30])
    pcolormesh_config.update(cmap='jet')
    # Overlay with the TEC map
    ipc = panel.overlay_pcolormesh(tec_, coords={'lat': glat, 'lon': glon, 'height': 250.}, cs='GEO',
                                    **pcolormesh_config)
    panel.add_colorbar(ipc, c_label="TECU", c_scale='linear', left=1.1, bottom=0.1, width=0.05, height=0.7)

    panel.add_title(title='APEX MLT-fixed')

    plt.savefig('manuscript_example_4_GNSS', dpi=200)
    plt.show()


if __name__ == "__main__":
    test_tec()
