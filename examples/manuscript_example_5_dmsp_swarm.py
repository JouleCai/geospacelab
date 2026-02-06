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
import pathlib as pathib

from geospacelab.visualization.mpl.dashboards import TSDashboard
import geospacelab.visualization.mpl.geomap.geodashboards as geomap

cwd = pathib.Path(__file__).parent.resolve()

def visual_dmsp_swarm(dmsp_dn, dmsp_sat_id, dmsp_orbit_id, swarm_dn, swarm_sat_id, pole='N'):
    
    # ds_swarm_pf = load_swarm_poynting_flux(swarm_dn, swarm_sat_id)
    
    band = 'LBHS'
    dt_fr = dmsp_dn - datetime.timedelta(minutes=60)
    dt_to = dmsp_dn + datetime.timedelta(minutes=60)
    time_c = dmsp_dn
    sat_id = dmsp_sat_id
    orbit_id = dmsp_orbit_id

    # Create a geodashboard object
    dashboard = geomap.GeoDashboard(dt_fr=dt_fr, dt_to=dt_to, figure_config={'figsize': (16, 14)})

    # If the orbit_id is specified, only one file will be downloaded. This option saves the downloading time.
    # dashboard.dock(datasource_contents=['jhuapl', 'dmsp', 'ssusi', 'edraur'], pole='N', sat_id='f17', orbit_id='46863')
    # If not specified, the data during the whole day will be downloaded.
    ds_dmsp_ssusi_edr = dashboard.dock(datasource_contents=['cdaweb', 'dmsp', 'ssusi', 'edr_aur'],
                                       pole=pole, sat_id=sat_id, orbit_id=orbit_id)
    ds_dmsp_ssusi_disk = dashboard.dock(datasource_contents=['cdaweb', 'dmsp', 'ssusi', 'sdr_disk'],
                                        pole=pole, sat_id=sat_id, orbit_id=orbit_id, pp_type='DAY_AURORAL')
    ds_s1 = dashboard.dock(
        datasource_contents=['madrigal', 'satellites', 'dmsp', 's1'],
        dt_fr=time_c - datetime.timedelta(minutes=45),
        dt_to=time_c + datetime.timedelta(minutes=45),
        sat_id=sat_id, replace_orbit=True)

    dt_fr_swarm = swarm_dn - datetime.timedelta(minutes=20)
    dt_to_swarm = swarm_dn + datetime.timedelta(minutes=20)
    ds_swarm = dashboard.dock(
        datasource_contents=['tud', 'swarm', 'dns_pod'],
        dt_fr=dt_fr_swarm,
        dt_to=dt_to_swarm,
        sat_id=swarm_sat_id, add_AACGM=True)

    dashboard.set_layout(1, 2, left=0.0, right=1.0, top=0.9, bottom=0.55, wspace=0)
    
    # Add the first polar map
    # Get the variables: LBHS emission intensiy, corresponding times and locations
    lbhs = ds_dmsp_ssusi_edr['GRID_AUR_' + band]
    dts = ds_dmsp_ssusi_edr['DATETIME']
    dts = dts.value.flatten()
    mlat = ds_dmsp_ssusi_edr['GRID_MLAT']
    mlon = ds_dmsp_ssusi_edr['GRID_MLON']
    mlt = ds_dmsp_ssusi_edr['GRID_MLT']

    # Search the index for the time to plot, used as an input to the following polar map
    ind_t = dashboard.datasets[1].get_time_ind(ut=time_c)
    if (dts[ind_t] - time_c).total_seconds()/60 > 60:     # in minutes
        raise ValueError("The time does not match any SSUSI data!")
    lbhs_ = lbhs.value[ind_t]
    mlat_ = mlat.value[ind_t]
    mlon_ = mlon.value[ind_t]
    mlt_ = mlt.value[ind_t]
    # Add a polar map panel to the dashboard. Currently the style is the fixed MLT at mlt_c=0. See the keywords below:
    panel = dashboard.add_polar_map(
        row_ind=0, col_ind=0, style='mlt-fixed', cs='AACGM',
        mlt_c=0., pole=pole, ut=time_c, boundary_lat=55., mirror_south=True
    )

    # Some settings for plotting.
    pcolormesh_config = lbhs.visual.plot_config.pcolormesh
    # Overlay the SSUSI image in the map.
    ipc = panel.overlay_pcolormesh(
        data=lbhs_, coords={'lat': mlat_, 'lon': mlon_, 'mlt': mlt_}, cs='AACGM', **pcolormesh_config)
    # Add a color bar
    panel.add_colorbar(ipc, c_label=band + " (R)", c_scale=pcolormesh_config['c_scale'], left=1.1, bottom=0.1,
                        width=0.05, height=0.7)

    # Overlay the gridlines
    panel.overlay_gridlines(lat_res=5, lon_label_separator=5)

    # Overlay the coastlines in the AACGM coordinate
    panel.overlay_coastlines()

    # Overlay cross-track velocity along satellite trajectory
    sc_dt = ds_s1['SC_DATETIME'].value.flatten()
    sc_lat = ds_s1['SC_GEO_LAT'].value.flatten()
    sc_lon = ds_s1['SC_GEO_LON'].value.flatten()
    sc_alt = ds_s1['SC_GEO_ALT'].value.flatten()
    sc_coords = {'lat': sc_lat, 'lon': sc_lon, 'height': sc_alt}

    v_H = ds_s1['v_i_H'].value.flatten()
    panel.overlay_cross_track_vector(
        vector=v_H, unit_vector=1000, vector_unit='m/s', alpha=0.3, color='red',
        sc_coords=sc_coords, sc_ut=sc_dt, cs='GEO',
    )
    # Overlay the satellite trajectory with ticks
    panel.overlay_sc_trajectory(sc_ut=sc_dt, sc_coords=sc_coords, cs='GEO')
    
    # Overlay swarm satellite trajectory
    sc2_dt = ds_swarm['SC_DATETIME'].value.flatten()
    sc2_lat = ds_swarm['SC_GEO_LAT'].value.flatten()
    sc2_lon = ds_swarm['SC_GEO_LON'].value.flatten()
    sc2_alt = ds_swarm['SC_GEO_ALT'].value.flatten()
    sc2_coords = {'lat': sc2_lat, 'lon': sc2_lon, 'height': sc2_alt}

    panel.overlay_sc_trajectory(sc_ut=sc2_dt, sc_coords=sc2_coords, cs='GEO', color='m')

    # Overlay sites
    panel.overlay_sites(site_ids=['TRO', 'ESR'], coords={'lat': [69.58, 78.15], 'lon': [19.23, 16.02], 'height': 0.}, cs='GEO', marker='^', markersize=5)

    # Add the title
    polestr = 'North' if pole == 'N' else 'South'
    panel.add_title(
            title='DMSP/SSUSI, EDR-AUR, ' + band + ', ' + sat_id.upper() + ', ' + polestr + ', ' + time_c.strftime('%Y-%m-%d %H%M UT'))

    # Add the second polar map on the right side
    # Get the variables: LBHS emission intensiy, corresponding times and locations
    lbhs = ds_dmsp_ssusi_disk['DISK_R_RECT_' + band]
    dts = ds_dmsp_ssusi_disk['DATETIME']
    scdt = ds_dmsp_ssusi_disk['SC_DATETIME']
    glat = ds_dmsp_ssusi_disk['DISK_GEO_LAT']
    glon = ds_dmsp_ssusi_disk['DISK_GEO_LON']
    alt = ds_dmsp_ssusi_disk['DISK_GEO_ALT']
    # mlt = dashboard.assign_variable(('GRID_MLT'), dataset_index=1).value

    # Search the index for the time to plot, used as an input to the following polar map
    ind_t = dashboard.datasets[1].get_time_ind(ut=time_c)
    if (dts.value[ind_t, 0] - time_c).total_seconds() / 60 > 60:  # in minutes
        raise ValueError("The time does not match any SSUSI data!")
    lbhs_ = lbhs.value[ind_t]
    lbhs_[lbhs_ < 1.] = 1.
    glat_ = glat.value[ind_t]
    glon_ = glon.value[ind_t]
    alt_ = alt.value[ind_t][0]

    glat_, glon_, lbhs_ = ds_dmsp_ssusi_disk.regriddata(disk_data=lbhs_, disk_geo_lat=glat_, disk_geo_lon=glon_)
    # Add a polar map panel to the dashboard. Currently the style is the fixed MLT at mlt_c=0. See the keywords below:
    panel = dashboard.add_polar_map(
        row_ind=0, col_ind=1, style='mlt-fixed', cs='AACGM',
        mlt_c=0., pole=pole, ut=time_c, boundary_lat=55., mirror_south=True
    )

    # Some settings for plotting.
    pcolormesh_config = lbhs.visual.plot_config.pcolormesh
    scdt_ = scdt.value[ind_t][:, np.newaxis]
    # Overlay the SSUSI image in the map.
    ipc = panel.overlay_pcolormesh(
        data=lbhs_, coords={'lat': glat_, 'lon': glon_, 'height': alt_}, ut=scdt_, cs='GEO', **pcolormesh_config,
        regridding=True, data_res=0.5, grid_res=0.05)
    # Add a color bar
    panel.add_colorbar(ipc, c_label=band + " (R)", c_scale=pcolormesh_config['c_scale'], left=1.1, bottom=0.1,
                        width=0.05, height=0.7)

    # Overlay the gridlines
    panel.overlay_gridlines(lat_res=5, lon_label_separator=5)

    # Overlay the coastlines in the AACGM coordinate
    panel.overlay_coastlines()

    # Overlay cross-track velocity along satellite trajectory
    sc_dt = ds_s1['SC_DATETIME'].value.flatten()
    sc_lat = ds_s1['SC_GEO_LAT'].value.flatten()
    sc_lon = ds_s1['SC_GEO_LON'].value.flatten()
    sc_alt = ds_s1['SC_GEO_ALT'].value.flatten()
    sc_coords = {'lat': sc_lat, 'lon': sc_lon, 'height': sc_alt}

    v_H = ds_s1['v_i_H'].value.flatten()
    panel.overlay_cross_track_vector(
        vector=v_H, unit_vector=1000, vector_unit='m/s', alpha=0.3, color='red',
        sc_coords=sc_coords, sc_ut=sc_dt, cs='GEO',
    )
    # Overlay the satellite trajectory with ticks
    panel.overlay_sc_trajectory(sc_ut=sc_dt, sc_coords=sc_coords, cs='GEO')

    # Overlay swarm satellite trajectory
    sc2_dt = ds_swarm['SC_DATETIME'].value.flatten()
    sc2_lat = ds_swarm['SC_GEO_LAT'].value.flatten()
    sc2_lon = ds_swarm['SC_GEO_LON'].value.flatten()
    sc2_alt = ds_swarm['SC_GEO_ALT'].value.flatten()
    sc2_coords = {'lat': sc2_lat, 'lon': sc2_lon, 'height': sc2_alt}

    panel.overlay_sc_trajectory(sc_ut=sc2_dt, sc_coords=sc2_coords, cs='GEO', color='m')

    # Overlay sites
    panel.overlay_sites(site_ids=['TRO', 'ESR'], coords={'lat': [69.58, 78.15], 'lon': [19.23, 16.02], 'height': 0.},
                         cs='GEO', marker='^', markersize=5)

    # Add the title
    polestr = 'North' if pole == 'N' else 'South'
    panel.add_title(
        title='DMSP/SSUSI, SDR-DISK, ' + band + ', ' + sat_id.upper() + ', ' + polestr + ', ' + time_c.strftime(
            '%Y-%m-%d %H%M UT'))

    # Lower panels
    # on the left side
    diff_minutes = 25
    dt_fr = dmsp_dn - datetime.timedelta(minutes=diff_minutes / 2)
    dt_to = dmsp_dn + datetime.timedelta(minutes=diff_minutes / 2)
    db_dmsp = TSDashboard(dt_fr=dt_fr, dt_to=dt_to, figure=dashboard.figure,
                              timeline_extra_labels=['GEO_LAT', 'GEO_LON', 'AACGM_LAT', 'AACGM_MLT'])

    dataset_s1 = db_dmsp.dock(datasource_contents=['madrigal', 'satellites', 'dmsp', 's1'], sat_id=sat_id)
    dataset_s4 = db_dmsp.dock(datasource_contents=['madrigal', 'satellites', 'dmsp', 's4'], sat_id=sat_id)
    dataset_e = db_dmsp.dock(datasource_contents=['madrigal', 'satellites', 'dmsp', 'e'], sat_id=sat_id)

    n_e = dataset_s1['n_e']
    v_i_H = dataset_s1['v_i_H']
    v_i_V = dataset_s1['v_i_V']
    d_B_D = dataset_s1['d_B_D']
    d_B_P = dataset_s1['d_B_P']
    d_B_F = dataset_s1['d_B_F']

    JE_e = dataset_e['JE_e']
    JE_i = dataset_e['JE_i']
    jE_e = dataset_e['jE_e']
    jE_i = dataset_e['jE_i']
    E_e_MEAN = dataset_e['E_e_MEAN']
    E_i_MEAN = dataset_e['E_i_MEAN']

    T_i = dataset_s4['T_i']
    T_e = dataset_s4['T_e']
    c_O_p = dataset_s4['COMP_O_p']

    layout = [
        [v_i_H, v_i_V],
        [d_B_P, d_B_D, d_B_F],
        [E_e_MEAN, E_i_MEAN],
        [JE_e, JE_i],
        [jE_e],
        [jE_i],
    ]
    db_dmsp.set_layout(panel_layouts=layout, left=0.1, right=0.45, top=0.5, hspace=0.0)
    db_dmsp.draw()
    # uts = db_dmsp.search_UTs(AACGM_LAT=66.6, GEO_LON=[8, 32])
    # if list(uts):
    #     db_dmsp.add_vertical_line(uts[0])
    # uts = db_dmsp.search_UTs(AACGM_LAT=75.4, GEO_LON=[0, 30])
    # if list(uts):
    #     db_dmsp.add_vertical_line(uts[0])
    db_dmsp.add_panel_labels()

    # On the right side
    delta_t = 12
    dt_fr = swarm_dn - datetime.timedelta(minutes=delta_t)
    dt_to = swarm_dn + datetime.timedelta(minutes=delta_t)

    timeline_extra_labels = ['GEO_LAT', 'GEO_LON', 'AACGM_LAT', 'AACGM_MLT']
    db_swarm = TSDashboard(dt_fr=dt_fr, dt_to=dt_to, timeline_extra_labels=timeline_extra_labels, figure=dashboard.figure)

    ds_swarm_tii = db_swarm.dock(
        datasource_contents=['esa_eo', 'swarm', 'advanced', 'efi_tct02'],
        product='TCT02', sat_id=swarm_sat_id, quality_control=False, add_AACGM=True
        )
    ds_swarm_lp = db_swarm.dock(
        datasource_contents=['esa_eo', 'swarm', 'advanced', 'efi_lp_hm'],
        product='LP_HM', sat_id='A', quality_control=False, add_AACGM=True
        )
    ds_swarm_lp_c = db_swarm.dock(
        datasource_contents=['esa_eo', 'swarm', 'advanced', 'efi_lp_hm'],
        product='LP_HM', sat_id='C', quality_control=False, add_AACGM=True
        )


    n_e = ds_swarm_lp['n_e']
    n_e_c = ds_swarm_lp_c['n_e']
    n_e.label = 'Swarm-A'
    n_e_c.label = 'Swarm-C'
    T_e = ds_swarm_lp['T_e']
    T_e_c = ds_swarm_lp_c['T_e']
    T_e.label = 'Swarm-A'
    T_e_c.label = 'Swarm-C'

    v_i_H_x = ds_swarm_tii['v_i_H_x']
    v_i_H_y = ds_swarm_tii['v_i_H_y']
    v_i_V_x = ds_swarm_tii['v_i_V_x']
    v_i_V_z = ds_swarm_tii['v_i_V_z']
     
    db_swarm.set_layout([[v_i_H_x, v_i_H_y, v_i_V_x, v_i_V_z], [n_e, n_e_c], [T_e, T_e_c], ],
                       left=0.58, right=0.93, top=0.5, hspace=0)
    db_swarm.draw()
    db_swarm.add_panel_labels()

    # db_swarm.show()
    db_swarm.save_figure(
        file_dir= cwd,
        file_name='manuscript_example_5_compare_v2_E' + swarm_dn.strftime('%Y-%m-%d') + '_SWARM-' + swarm_sat_id + '_DMSP-' + dmsp_sat_id.upper(), )


def event_1_1():
    dmsp_dn = datetime.datetime.strptime('20150317' + '125600', '%Y%m%d%H%M%S')
    dmsp_sat_id = 'f18'
    dmsp_orbit_id = '27899'
    
    swarm_dn = datetime.datetime.strptime('20150317' + '125600', '%Y%m%d%H%M%S')
    swarm_sat_id = 'A'
    
    visual_dmsp_swarm(dmsp_dn, dmsp_sat_id, dmsp_orbit_id, swarm_dn, swarm_sat_id, pole='N')


if __name__ == '__main__':
    event_1_1()
