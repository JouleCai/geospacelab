import datetime
import matplotlib.pyplot as plt

import geospacelab.visualization.mpl.geomap.geodashboards as geomap


def figure():
    db = geomap.GeoDashboard(dt_fr=None, dt_to=None, figure_config={'figsize': (25, 10)})
    db.set_layout(2, 4, hspace=0.2, top=0.95, bottom=0.05)

    band = 'LBHS'

    # Panel 1
    dt_fr = datetime.datetime(2015, 9, 30, 7, 0)
    dt_to = datetime.datetime(2015, 9, 30, 8, 0)
    time_1 = datetime.datetime(2015, 9, 30, 7, 37)
    pole = 'N'
    sat_id = 'f18'
    orbit_id = '30677'

    ds1 = db.dock(
        datasource_contents=['jhuapl', 'dmsp', 'ssusi', 'edraur'],
        dt_fr=dt_fr, dt_to=dt_to,
        pole=pole, sat_id=sat_id, orbit_id=orbit_id)

    lbhs = db.assign_variable('GRID_AUR_' + band, dataset=ds1)
    dts = db.assign_variable('DATETIME', dataset=ds1).value.flatten()
    mlat = db.assign_variable('GRID_MLAT', dataset=ds1).value
    mlon = db.assign_variable('GRID_MLON', dataset=ds1).value
    mlt = db.assign_variable(('GRID_MLT'), dataset=ds1).value

    panel1 = db.add_polar_map(row_ind=0, col_ind=0, style='mlt-fixed', cs='AACGM', mlt_c=0., pole=pole, ut=time_1,
                              boundary_lat=65., mirror_south=True)
    panel1.add_coastlines()
    pcolormesh_config = lbhs.visual.plot_config.pcolormesh
    pcolormesh_config.update(c_scale='log')
    pcolormesh_config.update(c_lim=[100, 2500])
    import geospacelab.visualization.mpl.colormaps as cm
    cmap = cm.cmap_gist_ncar_modified()
    cmap = 'viridis'
    pcolormesh_config.update(cmap=cmap)
    ipc = panel1.add_pcolor(lbhs.value[0], coords={'lat': mlat[0], 'lon': mlon[0], 'mlt': mlt[0], 'height': 250.},
                            cs='AACGM', **pcolormesh_config)
    # panel1.add_colorbar(ipc, c_label=band + " (R)", c_scale=pcolormesh_config['c_scale'], left=1.1, bottom=0.1,
    #                     width=0.05, height=0.7)

    panel1.add_gridlines(lat_res=5, lon_label_separator=5)

    pole_str = 'North' if pole == 'N' else 'South'
    panel1.add_title(title=time_1.strftime("%Y-%m-%dT%H:%M") + ', ' + pole_str + ', ' + sat_id.upper() + ', ' + 'Orbit: ' + orbit_id)
    panel1.add_label(y=0.95, label='(a)', fontsize=14)

    # Panel 2
    dt_fr = datetime.datetime(2015, 9, 30, 7, 0)
    dt_to = datetime.datetime(2015, 9, 30, 8, 0)
    time_1 = datetime.datetime(2015, 9, 30, 7, 58)
    pole = 'S'
    sat_id = 'f16'
    orbit_id = '61656'

    ds2 = db.dock(
        datasource_contents=['jhuapl', 'dmsp', 'ssusi', 'edraur'],
        dt_fr=dt_fr, dt_to=dt_to,
        pole=pole, sat_id=sat_id, orbit_id=orbit_id)

    lbhs = db.assign_variable('GRID_AUR_' + band, dataset=ds2)
    dts = db.assign_variable('DATETIME', dataset=ds2).value.flatten()
    mlat = db.assign_variable('GRID_MLAT', dataset=ds2).value
    mlon = db.assign_variable('GRID_MLON', dataset=ds2).value
    mlt = db.assign_variable(('GRID_MLT'), dataset=ds2).value

    panel2 = db.add_polar_map(row_ind=1, col_ind=0, style='mlt-fixed', cs='AACGM', mlt_c=0., pole=pole, ut=time_1,
                              boundary_lat=65., mirror_south=True)
    panel2.add_coastlines()
    pcolormesh_config = lbhs.visual.plot_config.pcolormesh
    pcolormesh_config.update(c_scale='log')
    pcolormesh_config.update(c_lim=[100, 2500])
    import geospacelab.visualization.mpl.colormaps as cm
    cmap = cm.cmap_gist_ncar_modified()
    cmap = 'viridis'
    pcolormesh_config.update(cmap=cmap)
    ipc = panel2.add_pcolor(lbhs.value[0], coords={'lat': mlat[0], 'lon': mlon[0], 'mlt': mlt[0], 'height': 250.},
                            cs='AACGM', **pcolormesh_config)
    # panel2.add_colorbar(ipc, c_label=band + " (R)", c_scale=pcolormesh_config['c_scale'], left=1.1, bottom=0.1,
    #                     width=0.05, height=0.7)

    panel2.add_gridlines(lat_res=5, lon_label_separator=5)

    pole_str = 'North' if pole == 'N' else 'South'
    panel2.add_title(title=time_1.strftime("%Y-%m-%dT%H:%M") + ', ' + pole_str + ', ' + sat_id.upper() + ', ' + 'Orbit: ' + orbit_id)
    panel2.add_label(y=0.95, label='(b)', fontsize=14)

    # Panel 3
    dt_fr = datetime.datetime(2015, 9, 3, 11, 0, 0)
    dt_to = datetime.datetime(2015, 9, 3, 11, 30, 0)
    time_1 = datetime.datetime(2015, 9, 3, 11, 19, 0)
    pole = 'N'
    sat_id = 'f16'
    orbit_id = '61277'

    ds3 = db.dock(
        datasource_contents=['jhuapl', 'dmsp', 'ssusi', 'edraur'],
        dt_fr=dt_fr, dt_to=dt_to,
        pole=pole, sat_id=sat_id, orbit_id=orbit_id)

    lbhs = db.assign_variable('GRID_AUR_' + band, dataset=ds3)
    dts = db.assign_variable('DATETIME', dataset=ds3).value.flatten()
    mlat = db.assign_variable('GRID_MLAT', dataset=ds3).value
    mlon = db.assign_variable('GRID_MLON', dataset=ds3).value
    mlt = db.assign_variable(('GRID_MLT'), dataset=ds3).value

    panel3 = db.add_polar_map(row_ind=0, col_ind=1, style='mlt-fixed', cs='AACGM', mlt_c=0., pole=pole, ut=time_1,
                              boundary_lat=65., mirror_south=True)
    panel3.add_coastlines()
    pcolormesh_config = lbhs.visual.plot_config.pcolormesh
    pcolormesh_config.update(c_scale='log')
    pcolormesh_config.update(c_lim=[100, 2500])
    import geospacelab.visualization.mpl.colormaps as cm
    cmap = cm.cmap_gist_ncar_modified()
    cmap = 'viridis'
    pcolormesh_config.update(cmap=cmap)
    ipc = panel3.add_pcolor(lbhs.value[0], coords={'lat': mlat[0], 'lon': mlon[0], 'mlt': mlt[0], 'height': 250.},
                            cs='AACGM', **pcolormesh_config)
    # panel2.add_colorbar(ipc, c_label=band + " (R)", c_scale=pcolormesh_config['c_scale'], left=1.1, bottom=0.1,
    #                     width=0.05, height=0.7)

    panel3.add_gridlines(lat_res=5, lon_label_separator=5)

    pole_str = 'North' if pole == 'N' else 'South'
    panel3.add_title(title=time_1.strftime("%Y-%m-%dT%H:%M") + ', ' + pole_str + ', ' + sat_id.upper() + ', ' + 'Orbit: ' + orbit_id)
    panel3.add_label(y=0.95, label='(c)', fontsize=14)

    # Panel 4
    dt_fr = datetime.datetime(2015, 9, 3, 12, 0, 0)
    dt_to = datetime.datetime(2015, 9, 3, 12, 30, 0)
    time_1 = datetime.datetime(2015, 9, 3, 12, 10, 0)
    pole = 'S'
    sat_id = 'f16'
    orbit_id = '61277'

    ds4 = db.dock(
        datasource_contents=['jhuapl', 'dmsp', 'ssusi', 'edraur'],
        dt_fr=dt_fr, dt_to=dt_to,
        pole=pole, sat_id=sat_id, orbit_id=orbit_id)

    lbhs = db.assign_variable('GRID_AUR_' + band, dataset=ds4)
    dts = db.assign_variable('DATETIME', dataset=ds4).value.flatten()
    mlat = db.assign_variable('GRID_MLAT', dataset=ds4).value
    mlon = db.assign_variable('GRID_MLON', dataset=ds4).value
    mlt = db.assign_variable(('GRID_MLT'), dataset=ds4).value

    panel4 = db.add_polar_map(row_ind=1, col_ind=1, style='mlt-fixed', cs='AACGM', mlt_c=0., pole=pole, ut=time_1,
                              boundary_lat=65., mirror_south=True)
    panel4.add_coastlines()
    pcolormesh_config = lbhs.visual.plot_config.pcolormesh
    pcolormesh_config.update(c_scale='log')
    pcolormesh_config.update(c_lim=[100, 2500])
    pcolormesh_config.update()
    import geospacelab.visualization.mpl.colormaps as cm
    cmap = cm.cmap_gist_ncar_modified()
    cmap = 'viridis'
    pcolormesh_config.update(cmap=cmap)
    ipc = panel4.add_pcolor(lbhs.value[0], coords={'lat': mlat[0], 'lon': mlon[0], 'mlt': mlt[0], 'height': 250.},
                            cs='AACGM', **pcolormesh_config)
    # panel2.add_colorbar(ipc, c_label=band + " (R)", c_scale=pcolormesh_config['c_scale'], left=1.1, bottom=0.1,
    #                     width=0.05, height=0.7)

    panel4.add_gridlines(lat_res=5, lon_label_separator=5)

    pole_str = 'North' if pole == 'N' else 'South'
    panel4.add_title(title=time_1.strftime("%Y-%m-%dT%H:%M") + ', ' + pole_str + ', ' + sat_id.upper() + ', ' + 'Orbit: ' + orbit_id)
    panel4.add_label(y=0.95, label='(d)', fontsize=14)
    #
    # Panel 5
    dt_fr = datetime.datetime(2015, 9, 27, 9, 0, 0)
    dt_to = datetime.datetime(2015, 9, 27, 9, 30, 0)
    time_1 = datetime.datetime(2015, 9, 27, 9, 26, 0)
    pole = 'N'
    sat_id = 'f16'
    orbit_id = '61615'

    dt_fr = datetime.datetime(2015, 9, 17, 18, 0, 0)
    dt_to = datetime.datetime(2015, 9, 17, 18, 30, 0)
    time_1 = datetime.datetime(2015, 9, 17, 18, 23, 0)
    pole = 'N'
    sat_id = 'f19'
    orbit_id = '07516'

    ds5 = db.dock(
        datasource_contents=['jhuapl', 'dmsp', 'ssusi', 'edraur'],
        dt_fr=dt_fr, dt_to=dt_to,
        pole=pole, sat_id=sat_id, orbit_id=orbit_id)

    lbhs = db.assign_variable('GRID_AUR_' + band, dataset=ds5)
    dts = db.assign_variable('DATETIME', dataset=ds5).value.flatten()
    mlat = db.assign_variable('GRID_MLAT', dataset=ds5).value
    mlon = db.assign_variable('GRID_MLON', dataset=ds5).value
    mlt = db.assign_variable(('GRID_MLT'), dataset=ds5).value

    panel5 = db.add_polar_map(row_ind=0, col_ind=2, style='mlt-fixed', cs='AACGM', mlt_c=0., pole=pole, ut=time_1,
                              boundary_lat=65., mirror_south=True)
    panel5.add_coastlines()
    pcolormesh_config = lbhs.visual.plot_config.pcolormesh
    pcolormesh_config.update(c_scale='log')
    pcolormesh_config.update(c_lim=[100, 2500])
    pcolormesh_config.update()
    import geospacelab.visualization.mpl.colormaps as cm
    cmap = cm.cmap_gist_ncar_modified()
    cmap = 'viridis'
    pcolormesh_config.update(cmap=cmap)
    ipc = panel5.add_pcolor(lbhs.value[0], coords={'lat': mlat[0], 'lon': mlon[0], 'mlt': mlt[0], 'height': 250.},
                            cs='AACGM', **pcolormesh_config)
    # panel2.add_colorbar(ipc, c_label=band + " (R)", c_scale=pcolormesh_config['c_scale'], left=1.1, bottom=0.1,
    #                     width=0.05, height=0.7)

    panel5.add_gridlines(lat_res=5, lon_label_separator=5)

    pole_str = 'North' if pole == 'N' else 'South'
    panel5.add_title(title=time_1.strftime("%Y-%m-%dT%H:%M") + ', ' + pole_str + ', ' + sat_id.upper() + ', ' + 'Orbit: ' + orbit_id)
    panel5.add_label(y=0.95, label='(e)', fontsize=14)

    # Panel 6
    dt_fr = datetime.datetime(2015, 9, 27, 10, 0, 0)
    dt_to = datetime.datetime(2015, 9, 27, 10, 30, 0)
    time_1 = datetime.datetime(2015, 9, 27, 10, 18, 0)
    pole = 'S'
    sat_id = 'f16'
    orbit_id = '61615'

    dt_fr = datetime.datetime(2015, 9, 17, 19, 0, 0)
    dt_to = datetime.datetime(2015, 9, 17, 19, 30, 0)
    time_1 = datetime.datetime(2015, 9, 17, 19, 23, 0)
    pole = 'S'
    sat_id = 'f16'
    orbit_id = '61479'

    ds6 = db.dock(
        datasource_contents=['jhuapl', 'dmsp', 'ssusi', 'edraur'],
        dt_fr=dt_fr, dt_to=dt_to,
        pole=pole, sat_id=sat_id, orbit_id=orbit_id)

    lbhs = db.assign_variable('GRID_AUR_' + band, dataset=ds6)
    dts = db.assign_variable('DATETIME', dataset=ds6).value.flatten()
    mlat = db.assign_variable('GRID_MLAT', dataset=ds6).value
    mlon = db.assign_variable('GRID_MLON', dataset=ds6).value
    mlt = db.assign_variable(('GRID_MLT'), dataset=ds6).value

    panel6 = db.add_polar_map(row_ind=1, col_ind=2, style='mlt-fixed', cs='AACGM', mlt_c=0., pole=pole, ut=time_1,
                              boundary_lat=65., mirror_south=True)
    panel6.add_coastlines()
    pcolormesh_config = lbhs.visual.plot_config.pcolormesh
    pcolormesh_config.update(c_scale='log')
    pcolormesh_config.update(c_lim=[100, 2500])
    pcolormesh_config.update()
    import geospacelab.visualization.mpl.colormaps as cm
    cmap = cm.cmap_gist_ncar_modified()
    cmap = 'viridis'
    pcolormesh_config.update(cmap=cmap)
    ipc = panel6.add_pcolor(lbhs.value[0], coords={'lat': mlat[0], 'lon': mlon[0], 'mlt': mlt[0], 'height': 250.},
                            cs='AACGM', **pcolormesh_config)
    # panel2.add_colorbar(ipc, c_label=band + " (R)", c_scale=pcolormesh_config['c_scale'], left=1.1, bottom=0.1,
    #                     width=0.05, height=0.7)

    panel6.add_gridlines(lat_res=5, lon_label_separator=5)

    pole_str = 'North' if pole == 'N' else 'South'
    panel6.add_title(title=time_1.strftime("%Y-%m-%dT%H:%M") + ', ' + pole_str + ', ' + sat_id.upper() + ', ' + 'Orbit: ' + orbit_id)
    panel6.add_label(y=0.95, label='(f)', fontsize=14)

    # Panel 7
    dt_fr = datetime.datetime(2015, 10, 10, 9, 0, 0)
    dt_to = datetime.datetime(2015, 10, 10, 10, 30, 0)
    time_1 = datetime.datetime(2015, 10, 10, 9, 58, 0)
    pole = 'N'
    sat_id = 'f16'
    orbit_id = '61799'

    ds7 = db.dock(
        datasource_contents=['jhuapl', 'dmsp', 'ssusi', 'edraur'],
        dt_fr=dt_fr, dt_to=dt_to,
        pole=pole, sat_id=sat_id, orbit_id=orbit_id)

    lbhs = db.assign_variable('GRID_AUR_' + band, dataset=ds7)
    dts = db.assign_variable('DATETIME', dataset=ds7).value.flatten()
    mlat = db.assign_variable('GRID_MLAT', dataset=ds7).value
    mlon = db.assign_variable('GRID_MLON', dataset=ds7).value
    mlt = db.assign_variable(('GRID_MLT'), dataset=ds7).value

    panel7 = db.add_polar_map(row_ind=0, col_ind=3, style='mlt-fixed', cs='AACGM', mlt_c=0., pole=pole, ut=time_1,
                              boundary_lat=65., mirror_south=True)
    panel7.add_coastlines()
    pcolormesh_config = lbhs.visual.plot_config.pcolormesh
    pcolormesh_config.update(c_scale='log')
    pcolormesh_config.update(c_lim=[100, 2500])
    pcolormesh_config.update()
    import geospacelab.visualization.mpl.colormaps as cm
    cmap = cm.cmap_gist_ncar_modified()
    cmap = 'viridis'
    pcolormesh_config.update(cmap=cmap)
    ipc = panel7.add_pcolor(lbhs.value[0], coords={'lat': mlat[0], 'lon': mlon[0], 'mlt': mlt[0], 'height': 250.},
                            cs='AACGM', **pcolormesh_config)
    # panel2.add_colorbar(ipc, c_label=band + " (R)", c_scale=pcolormesh_config['c_scale'], left=1.1, bottom=0.1,
    #                     width=0.05, height=0.7)

    panel7.add_gridlines(lat_res=5, lon_label_separator=5)

    pole_str = 'North' if pole == 'N' else 'South'
    panel7.add_title(title=time_1.strftime("%Y-%m-%dT%H:%M") + ', ' + pole_str + ', ' + sat_id.upper() + ', ' + 'Orbit: ' + orbit_id)
    panel7.add_label(y=0.95, label='(g)', fontsize=14)

    # Panel 8
    dt_fr = datetime.datetime(2015, 10, 10, 9, 0, 0)
    dt_to = datetime.datetime(2015, 10, 10, 10, 30, 0)
    time_1 = datetime.datetime(2015, 10, 10, 10, 50, 0)
    pole = 'S'
    sat_id = 'f16'
    orbit_id = '61799'

    ds8 = db.dock(
        datasource_contents=['jhuapl', 'dmsp', 'ssusi', 'edraur'],
        dt_fr=dt_fr, dt_to=dt_to,
        pole=pole, sat_id=sat_id, orbit_id=orbit_id)

    lbhs = db.assign_variable('GRID_AUR_' + band, dataset=ds8)
    dts = db.assign_variable('DATETIME', dataset=ds8).value.flatten()
    mlat = db.assign_variable('GRID_MLAT', dataset=ds8).value
    mlon = db.assign_variable('GRID_MLON', dataset=ds8).value
    mlt = db.assign_variable(('GRID_MLT'), dataset=ds8).value

    panel8 = db.add_polar_map(row_ind=1, col_ind=3, style='mlt-fixed', cs='AACGM', mlt_c=0., pole=pole, ut=time_1,
                              boundary_lat=65., mirror_south=True)
    panel8.add_coastlines()
    pcolormesh_config = lbhs.visual.plot_config.pcolormesh
    pcolormesh_config.update(c_scale='log')
    pcolormesh_config.update(c_lim=[100, 2500])
    pcolormesh_config.update()
    import geospacelab.visualization.mpl.colormaps as cm
    cmap = cm.cmap_gist_ncar_modified()
    cmap = 'viridis'
    pcolormesh_config.update(cmap=cmap)
    ipc = panel8.add_pcolor(lbhs.value[0], coords={'lat': mlat[0], 'lon': mlon[0], 'mlt': mlt[0], 'height': 250.},
                            cs='AACGM', **pcolormesh_config)
    panel8.add_colorbar(ipc, c_label=band + " (R)", c_scale=pcolormesh_config['c_scale'], left=1.2, bottom=0.1,
                         width=0.05, height=2)

    panel8.add_gridlines(lat_res=5, lon_label_separator=5)

    pole_str = 'North' if pole == 'N' else 'South'
    panel8.add_title(title=time_1.strftime("%Y-%m-%dT%H:%M") + ', ' + pole_str + ', ' + sat_id.upper() + ', ' + 'Orbit: ' + orbit_id)

    panel8.add_label(y=0.95, label='(h)', fontsize=14)
    plt.savefig('DMSP_SSUSI_', dpi=300)
    plt.show()


if __name__ == "__main__":
    figure()