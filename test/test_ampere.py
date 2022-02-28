import datetime
import matplotlib.pyplot as plt
import numpy as np

import geospacelab.visualization.mpl.geomap.geodashboards as geomap


def test_ampere():
    dt_fr = datetime.datetime(2016, 10, 23, 16)
    dt_to = datetime.datetime(2016, 10, 23, 23, 59)
    time1 = datetime.datetime(2016, 10, 23, 18, 10)
    pole = 'N'
    load_mode = 'assigned'
    # specify the file full path
    data_file_paths = ['/home/lei/afys-data/JHUAPL/AMPERE/Fitted/201610/20161023.0000.86400.120.north.grd.ncdf']

    viewer = geomap.GeoDashboard(dt_fr=dt_fr, dt_to=dt_to, figure_config={'figsize': (8, 8)})
    # viewer.dock(datasource_contents=['jhuapl', 'dmsp', 'ssusi', 'edraur'], pole='N', sat_id='f17', orbit_id='46863')
    viewer.dock(datasource_contents=['jhuapl', 'ampere', 'fitted'], load_mode=load_mode, data_file_paths=data_file_paths)
    viewer.set_layout(1, 1)
    dataset_ampere = viewer.datasets[1]

    fac = viewer.assign_variable('GRID_Jr', dataset_index=1)
    dts = viewer.assign_variable('DATETIME', dataset_index=1).value.flatten()
    mlat = viewer.assign_variable('GRID_MLAT', dataset_index=1).value
    mlt = viewer.assign_variable(('GRID_MLT'), dataset_index=1).value

    ind_t = dataset_ampere.get_time_ind(ut=time1)
    # initialize the polar map
    panel1 = viewer.add_polar_map(row_ind=0, col_ind=0, style='mlt-fixed', cs='AACGM', mlt_c=0., pole=pole, ut=time1, boundary_lat=60, mirror_south=True)

    panel1.overlay_coastlines()

    fac_ = fac.value[ind_t, :, :]
    # re-grid the original data with higher spatial resolution, default mlt_res = 0.05, mlat_res = 0.5. used for plotting.
    grid_mlat, grid_mlt, grid_fac = dataset_ampere.grid_fac(fac_, mlt_res=0.05, mlat_res=0.05, interp_method='linear')

    # remove values less than 0.2
    grid_fac[np.abs(grid_fac)<0.2] = np.nan
    # fac_[np.abs(fac_) < 0.2] = np.nan
    pcolormesh_config = fac.visual.plot_config.pcolormesh
    pcolormesh_config.update(c_scale='linear')
    pcolormesh_config.update(c_lim=[-1, 1])
    pcolormesh_config.update(shading='auto')
    import geospacelab.visualization.mpl.colormaps as cm
    cmap = cm.cmap_gist_ncar_modified()
    cmap = 'jet'
    pcolormesh_config.update(cmap=cmap)
    # ipc = panel1.add_pcolor(fac_, coords={'lat': mlat[ind_t, ::], 'lon': None, 'mlt': mlt[ind_t, ::], 'height': 250.}, cs='AACGM', **pcolormesh_config)
    ipm = panel1.overlay_pcolormesh(grid_fac, coords={'lat': grid_mlat, 'lon': None, 'mlt': grid_mlt, 'height': 250.}, cs='AACGM', **pcolormesh_config)
    panel1.add_colorbar(ipm, c_label=r'FAC ($\mu$A/m$^2$)', c_scale=pcolormesh_config['c_scale'], left=1.1, bottom=0.1,
                        width=0.05, height=0.7)

    panel1.overlay_gridlines(lat_res=5, lon_label_separator=5)

    polestr = 'North' if pole == 'N' else 'South'
    plt.savefig('ampere_example', dpi=300)
    plt.show()


if __name__ == "__main__":
    test_ampere()
