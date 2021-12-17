import datetime
import matplotlib.pyplot as plt

import geospacelab.visualization.mpl.geomap.geodashboards as geomap


def test_ssusi():
    dt_fr = datetime.datetime(2013, 3, 17, 0)
    dt_to = datetime.datetime(2013, 3, 19, 23, 59)
    time1 = datetime.datetime(2013, 3, 18, 17, 49)
    pole = 'N'
    sat_id = 'f18'
    band = 'LBHS'

    dashboard = geomap.GeoDashboard(dt_fr=dt_fr, dt_to=dt_to, figure_config={'figsize': (5, 5)})

    # dashboard.dock(datasource_contents=['jhuapl', 'dmsp', 'ssusi', 'edraur'], pole='N', sat_id='f17', orbit_id='46863')
    dashboard.dock(datasource_contents=['jhuapl', 'dmsp', 'ssusi', 'edraur'], pole=pole, sat_id=sat_id, orbit_id=None)
    dashboard.set_layout(1, 1)

    lbhs = dashboard.assign_variable('GRID_AUR_' + band, dataset_index=1)
    dts = dashboard.assign_variable('DATETIME', dataset_index=1).value.flatten()
    mlat = dashboard.assign_variable('GRID_MLAT', dataset_index=1).value
    mlon = dashboard.assign_variable('GRID_MLON', dataset_index=1).value
    mlt = dashboard.assign_variable(('GRID_MLT'), dataset_index=1).value

    for ind_t, dt in enumerate(dts):
        panel1 = dashboard.add_polar_map(row_ind=0, col_ind=0, style='mlt-fixed', cs='AACGM', mlt_c=0., pole=pole, ut=dt, boundary_lat=65., mirror_south=True)
        panel1.add_coastlines()

        lbhs_ = lbhs.value[ind_t, :, :]
        pcolormesh_config = lbhs.visual.plot_config.pcolormesh
        pcolormesh_config.update(c_scale='log')
        pcolormesh_config.update(c_lim=[100, 1500])
        import geospacelab.visualization.mpl.colormaps as cm
        cmap = cm.cmap_gist_ncar_modified()
        cmap = 'viridis'
        pcolormesh_config.update(cmap=cmap)
        ipc = panel1.add_pcolor(lbhs_, coords={'lat': mlat[ind_t, ::], 'lon': mlon[ind_t, ::], 'mlt': mlt[ind_t, ::], 'height': 250.}, cs='AACGM', **pcolormesh_config)
        panel1.add_colorbar(ipc, c_label=band + " (R)", c_scale=pcolormesh_config['c_scale'], left=1.1, bottom=0.1,
                            width=0.05, height=0.7)

        panel1.add_gridlines(lat_res=5, lon_label_separator=5)

        polestr = 'North' if pole == 'N' else 'South'
        panel1.add_title(title='DMSP/SSUSI, ' + band + ', ' + sat_id.upper() + ', ' + polestr + ', ' + dt.strftime('%Y-%m-%d %H%M UT'))
        plt.savefig('DMSP_SSUSI_' + dt.strftime('%Y%m%d-%H%M') + '_' + band + '_' + sat_id.upper() + '_' + pole, dpi=300)
        plt.clf()
        # plt.show()

    ind_t = dashboard.datasets[1].get_time_ind(ut=time1)

    panel1 = dashboard.add_polar_map(row_ind=0, col_ind=0, style='mlt-fixed', cs='AACGM', mlt_c=0., pole=pole, ut=time1, boundary_lat=65., mirror_south=True)
    # panel1 = dashboard.add_polar_map(row_ind=0, col_ind=0, style='mlt-fixed', cs='AACGM', mlt_c=0., pole=pole, ut=time1, mirror_south=True)
    # panel1 = dashboard.add_polar_map(row_ind=0, col_ind=0, style='lst-fixed', cs='GEO', lst_c=3., pole='N', ut=time1)
    # panel1 = dashboard.add_polar_map(row_ind=0, col_ind=0, style='lst-fixed', cs='GEO', lst_c=0, pole='S', ut=time1, mirror_south=True)
    # panel1 = dashboard.add_polar_map(row_ind=0, col_ind=0, style='lon-fixed', cs='GEO', lon_c=0., pole='S', ut=time1,
    #                           boundary_lat=0, mirror_south=False)
    # panel1 = dashboard.add_polar_map(row_ind=0, col_ind=0, style='lon-fixed', cs='GEO', lon_c=0., pole='N', ut=time1,
    #                          boundary_lat=0, mirror_south=False)
    panel1.add_coastlines()

    lbhs_ = lbhs.value[ind_t, :, :]
    pcolormesh_config = lbhs.visual.plot_config.pcolormesh
    pcolormesh_config.update(c_scale='log')
    pcolormesh_config.update(c_lim=[100, 1500])
    import geospacelab.visualization.mpl.colormaps as cm
    cmap = cm.cmap_gist_ncar_modified()
    cmap = 'viridis'
    pcolormesh_config.update(cmap=cmap)
    ipc = panel1.add_pcolor(lbhs_, coords={'lat': mlat[ind_t, ::], 'lon': mlon[ind_t, ::], 'mlt': mlt[ind_t, ::], 'height': 250.}, cs='AACGM', **pcolormesh_config)
    panel1.add_colorbar(ipc, c_label=band + " (R)", c_scale=pcolormesh_config['c_scale'], left=1.1, bottom=0.1,
                        width=0.05, height=0.7)

    panel1.add_gridlines(lat_res=5, lon_label_separator=5)

    polestr = 'North' if pole == 'N' else 'South'
    panel1.add_title(title='DMSP/SSUSI, ' + band + ', ' + sat_id.upper() + ', ' + polestr + ', ' + time1.strftime('%Y-%m-%d %H%M UT'))
    plt.savefig('DMSP_SSUSI_' + time1.strftime('%Y%m%d-%H%M') + '_' + band + '_' + sat_id.upper() + '_' + pole, dpi=300)
    plt.show()


if __name__ == "__main__":
    test_ssusi()
