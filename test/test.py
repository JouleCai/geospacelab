import datetime
import pathlib
import re
import matplotlib.pyplot as plt

import geospacelab.visualization.mpl.geomap.geodashboards as geomap
from geospacelab.visualization.mpl.dashboards import TSDashboard
import geospacelab.toolbox.utilities.pydatetime as dttool


def show_dmsp(sat_id=None, pole='N', band='LBHS', dt_c=None, file_dir=None):
    dt_fr = dttool.get_start_of_the_day(dt_c)
    dt_to = dttool.get_end_of_the_day(dt_c)

    # Create a geodashboard object
    dashboard = geomap.GeoDashboard(dt_fr=dt_fr, dt_to=dt_to, figure_config={'figsize': (16, 10)})

    # If the orbit_id is specified, only one file will be downloaded. This option saves the downloading time.
    # dashboard.dock(datasource_contents=['jhuapl', 'dmsp', 'ssusi', 'edraur'], pole='N', sat_id='f17', orbit_id='46863')
    # If not specified, the data during the whole day will be downloaded.
    dashboard.dock(datasource_contents=['jhuapl', 'dmsp', 'ssusi', 'edraur'], pole=pole, sat_id=sat_id, orbit_id=None)
    ds_s1 = dashboard.dock(
        datasource_contents=['madrigal', 'dmsp', 's1'],
        dt_fr=dt_c - datetime.timedelta(minutes=45),
        dt_to=dt_c + datetime.timedelta(minutes=45),
        sat_id=sat_id, replace_orbit=True)

    dashboard.set_layout(1, 1, left=0.05, right=0.35)

    # Get the variables: LBHS emission intensiy, corresponding times and locations
    lbhs = dashboard.assign_variable('GRID_AUR_' + band, dataset_index=1)
    dts = dashboard.assign_variable('DATETIME', dataset_index=1).value.flatten()
    mlat = dashboard.assign_variable('GRID_MLAT', dataset_index=1).value
    mlon = dashboard.assign_variable('GRID_MLON', dataset_index=1).value
    mlt = dashboard.assign_variable(('GRID_MLT'), dataset_index=1).value

    # Search the index for the time to plot, used as an input to the following polar map
    ind_t = dashboard.datasets[1].get_time_ind(ut=dt_c)
    if (dts[ind_t] - dt_c).total_seconds() / 60 > 60:  # in minutes
        raise ValueError("The time does not match any SSUSI data!")
    lbhs_ = lbhs.value[ind_t, :, :]
    mlat_ = mlat[ind_t, ::]
    mlon_ = mlon[ind_t, ::]
    mlt_ = mlt[ind_t, ::]
    # Add a polar map panel to the dashboard. Currently the style is the fixed MLT at mlt_c=0. See the keywords below:
    panel1 = dashboard.add_polar_map(
        row_ind=0, col_ind=0, style='mlt-fixed', cs='AACGM',
        mlt_c=0., pole=pole, ut=dt_c, boundary_lat=60., mirror_south=True
    )

    # Some settings for plotting.
    pcolormesh_config = lbhs.visual.plot_config.pcolormesh
    # Overlay the SSUSI image in the map.
    ipc = panel1.overlay_pcolormesh(
        data=lbhs_, coords={'lat': mlat_, 'lon': mlon_, 'mlt': mlt_}, cs='AACGM', **pcolormesh_config)
    # Add a color bar
    panel1.add_colorbar(ipc, c_label=band + " (R)", c_scale=pcolormesh_config['c_scale'], left=1.1, bottom=0.1,
                        width=0.05, height=0.7)

    # Overlay the gridlines
    panel1.overlay_gridlines(lat_res=5, lon_label_separator=5)

    # Overlay the coastlines in the AACGM coordinate
    panel1.overlay_coastlines()

    # Overlay cross-track velocity along satellite trajectory
    sc_dt = ds_s1['SC_DATETIME'].value.flatten()
    sc_lat = ds_s1['SC_GEO_LAT'].value.flatten()
    sc_lon = ds_s1['SC_GEO_LON'].value.flatten()
    sc_alt = ds_s1['SC_GEO_ALT'].value.flatten()
    sc_coords = {'lat': sc_lat, 'lon': sc_lon, 'height': sc_alt}

    v_H = ds_s1['v_i_H'].value.flatten()
    panel1.overlay_cross_track_vector(
        vector=v_H, unit_vector=1000, vector_unit='m/s', alpha=0.3, color='red',
        sc_coords=sc_coords, sc_ut=sc_dt, cs='GEO',
    )
    # Overlay the satellite trajectory with ticks
    panel1.overlay_sc_trajectory(sc_ut=sc_dt, sc_coords=sc_coords, cs='GEO')

    # Overlay sites
    panel1.overlay_sites(site_ids=['TRO', 'ESR'],
                         coords={'lat': [69.58, 78.15], 'lon': [19.23, 16.02], 'height': [0, 0]}, cs='GEOC', marker='^',
                         markersize=2)

    # Add the title and save the figure
    polestr = 'North' if pole == 'N' else 'South'
    panel1.add_title(
        title='DMSP/SSUSI, ' + band + ', ' + sat_id.upper() + ', ' + polestr + ', ' + dt_c.strftime('%Y-%m-%d %H%M UT'))

    # Add TSDashboard
    diff_minutes = 5
    dt_fr_2 = dt_c - datetime.timedelta(minutes=diff_minutes / 2)
    dt_to_2 = dt_c + datetime.timedelta(minutes=diff_minutes / 2)
    dashboard_2 = TSDashboard(dt_fr=dt_fr_2, dt_to=dt_to_2, figure=dashboard.figure,
                              timeline_extra_labels=['GEO_LAT', 'GEO_LON', 'AACGM_LAT', 'AACGM_MLT'])

    dataset_s1 = dashboard_2.dock(datasource_contents=['madrigal', 'dmsp', 's1'], sat_id=sat_id)
    # dataset_s4 = dashboard_2.dock(datasource_contents=['madrigal', 'dmsp', 's4'], sat_id=sat_id)
    dataset_e = dashboard_2.dock(datasource_contents=['madrigal', 'dmsp', 'e'], sat_id=sat_id)

    n_e = dashboard_2.assign_variable('n_e', dataset=dataset_s1)
    v_i_H = dashboard_2.assign_variable('v_i_H', dataset=dataset_s1)
    v_i_V = dashboard_2.assign_variable('v_i_V', dataset=dataset_s1)
    d_B_D = dashboard_2.assign_variable('d_B_D', dataset=dataset_s1)
    d_B_P = dashboard_2.assign_variable('d_B_P', dataset=dataset_s1)
    d_B_F = dashboard_2.assign_variable('d_B_F', dataset=dataset_s1)

    JE_e = dashboard_2.assign_variable('JE_e', dataset=dataset_e)
    JE_i = dashboard_2.assign_variable('JE_i', dataset=dataset_e)
    jE_e = dashboard_2.assign_variable('jE_e', dataset=dataset_e)
    jE_i = dashboard_2.assign_variable('jE_i', dataset=dataset_e)
    E_e_MEAN = dashboard_2.assign_variable('E_e_MEAN', dataset=dataset_e)
    E_i_MEAN = dashboard_2.assign_variable('E_i_MEAN', dataset=dataset_e)

    #    T_i = dashboard_2.assign_variable('T_i', dataset=dataset_s4)
    #    T_e = dashboard_2.assign_variable('T_e', dataset=dataset_s4)
    #    c_O_p = dashboard_2.assign_variable('COMP_O_p', dataset=dataset_s4)

    layout = [
        [v_i_H, v_i_V],
        [d_B_P, d_B_D, d_B_F],
        [E_e_MEAN, E_i_MEAN],
        [JE_e, JE_i],
        [jE_e],
        [jE_i],
    ]
    dashboard_2.set_layout(panel_layouts=layout, left=0.5, right=0.9, hspace=0.01)
    dashboard_2.draw()
    uts = dashboard_2.search_UTs(AACGM_LAT=66.6, GEO_LON=[0, 40])
    if list(uts):
        dashboard_2.add_vertical_line(uts[0])
    uts = dashboard_2.search_UTs(AACGM_LAT=75.4, GEO_LON=[0, 40])
    if list(uts):
        dashboard_2.add_vertical_line(uts[0])
    dashboard_2.add_panel_labels()

    file_name = 'DMSP_' + dt_c.strftime(
        '%Y-%m-%d_%H%M') + '_' + sat_id.upper() + '_SSUSI_' + band + '_SSJ_SSM_SSIES_' + pole
    file_path_out = file_dir / file_name
    plt.savefig(file_path_out, dpi=300)
    print(f'file out:{file_path_out}')
    # show the figure
    # plt.show()


def search_record():
    root_dir = pathlib.Path("/home/lei/01-Work/01-Project/OY21-VisitingPhD/events/good_cases")
    root_dir = pathlib.Path("/home/lei/01-Work/01-Project/OY21-VisitingPhD/events/good_cases_waitlist")
    file_paths = root_dir.glob("**/*conjugation*.png")
    sat_ids = []
    dts = []
    file_dirs = []
    for ind, fp in enumerate(file_paths):
        fn = fp.stem
        print(fp)
        rc = re.compile("(F[\d]+)[\w\s]+([\d]{8}\s[\d]{6})")
        result = rc.findall(fn)
        if not list(result):
            continue
        sat_id = result[0][0]
        dt = datetime.datetime.strptime(result[0][1], "%Y%m%d %H%M%S")
        file_dirs.append(fp.parent.resolve())
        sat_ids.append(sat_id.lower())
        dts.append(dt)

    return file_dirs, sat_ids, dts


def good_events():
    file_dirs, sat_ids, dts = search_record()

    falsed_events = []
    for i, (fd, sat_id, dt) in enumerate(zip(file_dirs, sat_ids, dts)):
        try:
            show_dmsp(sat_id=sat_id, dt_c=dt, file_dir=fd)
        except:
            falsed_events.append([dt, fd, sat_id])
        # show_dmsp(sat_id=sat_id, dt_c=dt, file_dir=fd)

    for info in falsed_events:
        print(info)


if __name__ == "__main__":
    # good_events()

    sat_id = 'f16'
    dt = datetime.datetime.strptime('20160314 2305', "%Y%m%d %H%M%S")
    file_dir = pathlib.Path(__file__).parent.resolve()
    show_dmsp(sat_id=sat_id, dt_c=dt, file_dir=file_dir)
