import datetime
import matplotlib.pyplot as plt

import geospacelab.visualization.mpl.geomap.geodashboards as geomap


def test_ssusi():
    dt_fr = datetime.datetime(2015, 9, 8, 8)
    dt_to = datetime.datetime(2015, 9, 8, 23, 59)
    time1 = datetime.datetime(2015, 9, 8, 20, 21)
    pole = 'N'
    sat_id = 'f16'
    band = 'LBHS'

    # Create a geodashboard object
    dashboard = geomap.GeoDashboard(dt_fr=dt_fr, dt_to=dt_to, figure_config={'figsize': (5, 5)})

    # If the orbit_id is specified, only one file will be downloaded. This option saves the downloading time.
    # dashboard.dock(datasource_contents=['jhuapl', 'dmsp', 'ssusi', 'edraur'], pole='N', sat_id='f17', orbit_id='46863')
    # If not specified, the data during the whole day will be downloaded.
    dashboard.dock(datasource_contents=['jhuapl', 'dmsp', 'ssusi', 'edraur'], pole=pole, sat_id=sat_id, orbit_id=None)
    dashboard.set_layout(1, 1)

    # Get the variables: LBHS emission intensiy, corresponding times and locations
    lbhs = dashboard.assign_variable('GRID_AUR_' + band, dataset_index=1)
    dts = dashboard.assign_variable('DATETIME', dataset_index=1).value.flatten()
    mlat = dashboard.assign_variable('GRID_MLAT', dataset_index=1).value
    mlon = dashboard.assign_variable('GRID_MLON', dataset_index=1).value
    mlt = dashboard.assign_variable(('GRID_MLT'), dataset_index=1).value

    # Search the index for the time to plot, used as an input to the following polar map
    ind_t = dashboard.datasets[1].get_time_ind(ut=time1)

    # Add a polar map panel to the dashboard. Currently the style is the fixed MLT at mlt_c=0. See the keywords below:
    panel1 = dashboard.add_polar_map(row_ind=0, col_ind=0, style='mlt-fixed', cs='AACGM', mlt_c=0., pole=pole, ut=time1, boundary_lat=65., mirror_south=True)
    # panel1 = dashboard.add_polar_map(row_ind=0, col_ind=0, style='mlt-fixed', cs='AACGM', mlt_c=0., pole=pole, ut=time1, mirror_south=True)
    # panel1 = dashboard.add_polar_map(row_ind=0, col_ind=0, style='lst-fixed', cs='GEO', lst_c=3., pole='N', ut=time1)
    # panel1 = dashboard.add_polar_map(row_ind=0, col_ind=0, style='lst-fixed', cs='GEO', lst_c=0, pole='S', ut=time1, mirror_south=True)
    # panel1 = dashboard.add_polar_map(row_ind=0, col_ind=0, style='lon-fixed', cs='GEO', lon_c=0., pole='S', ut=time1,
    #                           boundary_lat=0, mirror_south=False)
    # panel1 = dashboard.add_polar_map(row_ind=0, col_ind=0, style='lon-fixed', cs='GEO', lon_c=0., pole='N', ut=time1,
    #                          boundary_lat=0, mirror_south=False)

    # Add the coastlines in the AACGM coordinate
    panel1.add_coastlines()

    # Some settings for plotting.
    lbhs_ = lbhs.value[ind_t, :, :]
    pcolormesh_config = lbhs.visual.plot_config.pcolormesh
    pcolormesh_config.update(c_scale='log')
    pcolormesh_config.update(c_lim=[100, 1500])
    import geospacelab.visualization.mpl.colormaps as cm
    cmap = cm.cmap_gist_ncar_modified()
    cmap = 'viridis'
    pcolormesh_config.update(cmap=cmap)
    # Overlay the SSUSI image in the map.
    ipc = panel1.add_pcolor(lbhs_, coords={'lat': mlat[ind_t, ::], 'lon': mlon[ind_t, ::], 'mlt': mlt[ind_t, ::], 'height': 250.}, cs='AACGM', **pcolormesh_config)
    # Add a color bar
    panel1.add_colorbar(ipc, c_label=band + " (R)", c_scale=pcolormesh_config['c_scale'], left=1.1, bottom=0.1,
                        width=0.05, height=0.7)

    # Add the gridlines
    panel1.add_gridlines(lat_res=5, lon_label_separator=5)

    # Add the title and save the figure
    polestr = 'North' if pole == 'N' else 'South'
    panel1.add_title(title='DMSP/SSUSI, ' + band + ', ' + sat_id.upper() + ', ' + polestr + ', ' + time1.strftime('%Y-%m-%d %H%M UT'))
    plt.savefig('DMSP_SSUSI_' + time1.strftime('%Y%m%d-%H%M') + '_' + band + '_' + sat_id.upper() + '_' + pole, dpi=300)

    # show the figure
    plt.show()


if __name__ == "__main__":
    test_ssusi()
