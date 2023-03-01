import datetime
import matplotlib.pyplot as plt
import numpy as np

import geospacelab.visualization.mpl.geomap.geodashboards as geomap


def test_ampere():
    dt_fr = datetime.datetime(2016, 3, 15, 0)
    dt_to = datetime.datetime(2016, 3, 15, 23, 59)
    time1 = datetime.datetime(2016, 3, 15, 1, 10)
    pole = 'N'
    load_mode = 'assigned'
    # specify the file full path
    data_file_paths = ['/home/lei/afys-data/SuperDARN/PotentialMap/2016/new.dat']
    # data_file_paths = ['/Users/lcai/Geospacelab/Data/SuperDARN/POTMAP/2016/SuperDARM_POTMAP_20160314_10min_test.txt']

    viewer = geomap.GeoDashboard(dt_fr=dt_fr, dt_to=dt_to, figure_config={'figsize': (8, 8)})
    viewer.dock(datasource_contents=['superdarn', 'potmap'], load_mode=load_mode, data_file_paths=data_file_paths)
    viewer.set_layout(1, 1)
    dataset_superdarn = viewer.datasets[1]

    phi = viewer.assign_variable('GRID_phi', dataset_index=1)
    dts = viewer.assign_variable('DATETIME', dataset_index=1).value.flatten()
    mlat = viewer.assign_variable('GRID_MLAT', dataset_index=1)
    mlon = viewer.assign_variable('GRID_MLON', dataset_index=1)
    mlt = viewer.assign_variable(('GRID_MLT'), dataset_index=1)

    ind_t = dataset_superdarn.get_time_ind(ut=time1)
    # initialize the polar map
    panel1 = viewer.add_polar_map(row_ind=0, col_ind=0, style='mlt-fixed', cs='AACGM', mlt_c=0., pole=pole, ut=time1, boundary_lat=50, mirror_south=True)

    panel1.overlay_coastlines()

    phi_ = phi.value[ind_t]
    mlat_ = mlat.value[ind_t]
    mlt_ = mlt.value[ind_t]
    mlon_ = mlon.value[ind_t]

    # grid_mlat, grid_mlt, grid_phi = dataset_superdarn.grid_phi(mlat_, mlt_, phi_, interp_method='cubic')
    grid_mlat, grid_mlt, grid_phi = dataset_superdarn.postprocess_roll(mlat_, mlt_, phi_)

    # re-grid the original data with higher spatial resolution, default mlt_res = 0.05, mlat_res = 0.5. used for plotting.
    # grid_mlat, grid_mlt, grid_fac = dataset_ampere.grid_fac(phi_, mlt_res=0.05, mlat_res=0.05, interp_method='linear')

    levels = np.array([-21e3, -18e3, -15e3, -12e3, -9e3, -6e3, 3e3, 6e3,  9e3, 12e3, 15e3, 18e3, 21e3])
    # ipc = panel1.add_pcolor(fac_, coords={'lat': mlat[ind_t, ::], 'lon': None, 'mlt': mlt[ind_t, ::], 'height': 250.}, cs='AACGM', **pcolormesh_config)
    ict = panel1.overlay_contour(grid_phi, coords={'lat': grid_mlat, 'lon': None, 'mlt': grid_mlt}, cs='AACGM', colors='b', levels=levels)
    # panel1.major_ax.clabel(ict, inline=True, fontsize=10)
    panel1.overlay_gridlines(lat_res=5, lon_label_separator=5)

    polestr = 'North' if pole == 'N' else 'South'
    # panel1.add_title('DMSP/SSUSI, ' + band + ', ' + sat_id.upper() + ', ' + polestr + ', ' + time1.strftime('%Y-%m-%d %H%M UT'), pad=20)
    plt.savefig('superdarn_example', dpi=300)
    plt.show()


if __name__ == "__main__":
    test_ampere()
