import datetime
import matplotlib.pyplot as plt
import pathlib
import re

import geospacelab.visualization.mpl.geomap.geodashboards as geomap
import geospacelab.toolbox.utilities.pydatetime as dttool


def show_dmsp(sat_id=None, pole='N', band='LBHS', dt_c=None, file_dir=None):
    dt_fr = dttool.get_start_of_the_day(dt_c)
    dt_to = dttool.get_end_of_the_day(dt_c)

    # Create a geodashboard object
    dashboard = geomap.GeoDashboard(dt_fr=dt_fr, dt_to=dt_to, figure_config={'figsize': (5, 5)})

    # If the orbit_id is specified, only one file will be downloaded. This option saves the downloading time.
    # dashboard.dock(datasource_contents=['jhuapl', 'dmsp', 'ssusi', 'edraur'], pole='N', sat_id='f17', orbit_id='46863')
    # If not specified, the data during the whole day will be downloaded.
    dashboard.dock(datasource_contents=['jhuapl', 'dmsp', 'ssusi', 'edraur'], pole=pole, sat_id=sat_id, orbit_id=None)
    ds_s1 = dashboard.dock(
        datasource_contents=['madrigal', 'dmsp', 's1'],
        dt_fr=dt_c - datetime.timedelta(minutes=45),
        dt_to=dt_c + datetime.timedelta(minutes=45),
        sat_id=sat_id)

    dashboard.set_layout(1, 1)

    # Get the variables: LBHS emission intensiy, corresponding times and locations
    lbhs = dashboard.assign_variable('GRID_AUR_' + band, dataset_index=1)
    dts = dashboard.assign_variable('DATETIME', dataset_index=1).value.flatten()
    mlat = dashboard.assign_variable('GRID_MLAT', dataset_index=1).value
    mlon = dashboard.assign_variable('GRID_MLON', dataset_index=1).value
    mlt = dashboard.assign_variable(('GRID_MLT'), dataset_index=1).value

    # Search the index for the time to plot, used as an input to the following polar map
    ind_t = dashboard.datasets[1].get_time_ind(ut=dt_c)
    lbhs_ = lbhs.value[ind_t, :, :]
    mlat_ = mlat[ind_t, ::]
    mlon_ = mlon[ind_t, ::]
    mlt_ = mlt[ind_t, ::]
    # Add a polar map panel to the dashboard. Currently the style is the fixed MLT at mlt_c=0. See the keywords below:
    panel1 = dashboard.add_polar_map(row_ind=0, col_ind=0, style='mlt-fixed', cs='AACGM', mlt_c=0., pole=pole, ut=dt_c,
                                     boundary_lat=65., mirror_south=True)

    # Some settings for plotting.
    pcolormesh_config = lbhs.visual.plot_config.pcolormesh
    # Overlay the SSUSI image in the map.
    ipc = panel1.overlay_pcolormesh(data=lbhs_, coords={'lat': mlat_, 'lon': mlon_, 'mlt': mlt_}, cs='AACGM',
                                    **pcolormesh_config)
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
    panel1.overlay_cross_track_vector(vector=v_H, unit_vector=1000, alpha=0.5, color='r', sc_coords=sc_coords,
                                      sc_ut=sc_dt)
    # Overlay the satellite trajectory with ticks
    panel1.overlay_sc_trajectory(sc_ut=sc_dt, sc_coords=sc_coords, cs='GEO')

    # Add the title and save the figure
    polestr = 'North' if pole == 'N' else 'South'
    panel1.add_title(
        title='DMSP/SSUSI, ' + band + ', ' + sat_id.upper() + ', ' + polestr + ', ' + dt_c.strftime('%Y-%m-%d %H%M UT'))

    plt.savefig('DMSP_SSUSI_' + dt_c.strftime('%Y%m%d-%H%M') + '_' + band + '_' + sat_id.upper() + '_' + pole, dpi=300)

    # show the figure
    plt.show()


def search_record():
    root_dir = pathlib.Path("/home/lei/01-Work/01-Project/OY21-VisitingPhD/events/good_cases")
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
        sat_ids.append(sat_id)
        dts.append(dt)

    return file_dirs, sat_ids, dts


def good_events():
    file_dirs, sat_ids, dts = search_record()

    for i, (fd, sat_id, dt) in enumerate(zip(file_dirs, sat_ids, dts)):
        try:
            show_dmsp(sat_id=sat_id, dt_c=dt, file_dir=fd)
        except:
            print(fd, sat_id, dt)


if __name__ == "__main__":
    good_events()
