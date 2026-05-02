import datetime
import matplotlib.pyplot as plt
import numpy as np
import pathlib

import geospacelab.visualization.mpl.dashboards as dashboards

cwd = pathlib.Path(__file__).parent.resolve()
file_dir_figure = cwd / 'figures'
file_dir_figure.mkdir(parents=True, exist_ok=True)

def test_swarm_EFI_LP_1B_overview():
    """Test Swarm EFI/LP 1B data product
    
    """
    dt_fr = datetime.datetime(2016, 3, 14, 20, 15)
    dt_to = datetime.datetime(2016, 3, 14, 21, 0)

    db = dashboards.TSDashboard(
        dt_fr=dt_fr, dt_to=dt_to, figure_config={'figsize': (8, 10)},
        timeline_extra_labels=['GEO_LAT', 'GEO_LON', 'APEX_LAT', 'APEX_LON', 'APEX_MLT'],
        )

    ds = db.dock(
        datasource_contents=['esa_eo', 'swarm', 'l1b', 'efi_lp'], 
        product_version='latest', # 'latest' (default),
        sat_id='A', add_APEX=True)
    
    panel_layouts = [
        [ds['n_i'], ds['n_e']],
        [ds['T_e']],
        [ds['V_SC']],
        [ds['FLAG_LP'], ds['FLAG_n_i'], ds['FLAG_n_e'], ds['FLAG_T_e'], ds['FLAG_V_SC']],
        [ds['FLAG_1_BIN_AUX'],],
        [ds['FLAG_2_BIN_AUX'],],
    ]
    # Change some default attrs for better visualization
    ds['T_e'].visual.axis[1].lim = [-10, 10000]

    db.set_layout(panel_layouts=panel_layouts)
    db.draw()
    db.add_title(title='Swarm-{} EFI/LP 1B Overview'.format(ds.sat_id), fontsize='medium', append_time=True)

    db.save_figure(file_dir=file_dir_figure, file_name='example_EFI_LP_1B_Swarm-{}_overview'.format(ds.sat_id), dpi=100, append_time=False)
    db.show()


def test_swarm_EFI_LPI_1B_overview():
    """Test Swarm EFI/LPI 1B data product
    
    """
    dt_fr = datetime.datetime(2016, 3, 14, 20, 15)
    dt_to = datetime.datetime(2016, 3, 14, 21, 0)

    db = dashboards.TSDashboard(
        dt_fr=dt_fr, dt_to=dt_to, figure_config={'figsize': (8, 10)},
        timeline_extra_labels=['GEO_LAT', 'GEO_LON', 'APEX_LAT', 'APEX_LON', 'APEX_MLT'],
        )

    ds = db.dock(
        datasource_contents=['esa_eo', 'swarm', 'l1b', 'efi_lpi'], 
        product_version='latest', # 'latest' (default),
        sat_id='A', add_APEX=True)
    
    panel_layouts = [
        [ds['n_i'], ds['n_e']],
        [ds['T_e']],
        [ds['V_SC']],
        [ds['FLAG_LP'], ds['FLAG_n_i'], ds['FLAG_n_e'], ds['FLAG_T_e'], ds['FLAG_V_SC']],
        [ds['FLAG_1_BIN_AUX'],],
        [ds['FLAG_2_BIN_AUX'],],
    ]
    # Change some default attrs for better visualization
    ds['T_e'].visual.axis[1].lim = [-10, 10000]

    db.set_layout(panel_layouts=panel_layouts)
    db.draw()
    db.add_title(title='Swarm-{} EFI/LPI 1B Overview'.format(ds.sat_id), fontsize='medium', append_time=True)

    db.save_figure(file_dir=file_dir_figure, file_name='example_EFI_LPI_1B_Swarm-{}_overview'.format(ds.sat_id), dpi=100, append_time=False)
    db.show()
    

def test_swarm_EFI_LP_FP_overview():
    """Test Swarm EFI/LP-FP data product
    
    """
    dt_fr = datetime.datetime(2016, 3, 14, 20, 15)
    dt_to = datetime.datetime(2016, 3, 14, 21, 0)

    db = dashboards.TSDashboard(
        dt_fr=dt_fr, dt_to=dt_to, figure_config={'figsize': (8, 10)},
        timeline_extra_labels=['GEO_LAT', 'GEO_LON', 'APEX_LAT', 'APEX_LON', 'APEX_MLT'],
        )

    ds = db.dock(
        datasource_contents=['esa_eo', 'swarm', 'advanced', 'efi_lp_fp'], 
        product_version='latest', # 'latest' (default),
        sat_id='A', add_APEX=True)
    
    panel_layouts = [
        [ds['n_p']],
        [ds['I_FP']],
    ]
    
    db.set_layout(panel_layouts=panel_layouts)
    db.draw()
    db.add_title(title='Swarm-{} EFI/LP-FP Overview'.format(ds.sat_id), fontsize='medium', append_time=True)

    db.save_figure(file_dir=file_dir_figure, file_name='example_EFI_LP_FP_Swarm-{}_overview'.format(ds.sat_id), dpi=100, append_time=False)
    db.show()
    

def test_swarm_compare_EFI_LP_FP():
    """Test comparison of Swarm EFI/LP-FP data product with EFI/LP 1B data product
    
    """
    dt_fr = datetime.datetime(2016, 3, 14, 20, 15)
    dt_to = datetime.datetime(2016, 3, 14, 20, 35)

    db = dashboards.TSDashboard(
        dt_fr=dt_fr, dt_to=dt_to, figure_config={'figsize': (8, 10)},
        timeline_extra_labels=['GEO_LAT', 'GEO_LON', 'APEX_LAT', 'APEX_LON', 'APEX_MLT'],
        )

    ds_lp_fp = db.dock(
        datasource_contents=['esa_eo', 'swarm', 'advanced', 'efi_lp_fp'], 
        product_version='latest', # 'latest' (default),
        sat_id='A', add_APEX=True)
    
    ds_lp_1b = db.dock(
        datasource_contents=['esa_eo', 'swarm', 'l1b', 'efi_lp'], 
        product_version='latest', # 'latest' (default),
        sat_id='A', add_APEX=True)
    
    panel_layouts = [
        [ds_lp_fp['n_p'], ds_lp_1b['n_i'], ds_lp_1b['n_e']],
        [ds_lp_1b['T_e']],
        [ds_lp_1b['FLAG_LP'], ds_lp_1b['FLAG_n_i'], ds_lp_1b['FLAG_n_e'], ds_lp_1b['FLAG_T_e'], ds_lp_1b['FLAG_V_SC']],
        [ds_lp_1b['FLAG_1_BIN_AUX'],],
        [ds_lp_1b['FLAG_2_BIN_AUX'],],
    ]
    ds_lp_1b['n_i'].visual.axis[2].label = r"n$_i$ (1B)"
    ds_lp_1b['n_e'].visual.axis[2].label = r"n$_e$ (1B)"
    ds_lp_1b['T_e'].visual.axis[2].label = r"T$_e$ (1B)"
    ds_lp_fp['n_p'].visual.axis[2].label = r"n$_p$ (FP)"
    
    db.set_layout(panel_layouts=panel_layouts)
    db.draw()
    db.add_title(title='Swarm-{} EFI/LP-FP vs. EFI/LP 1B Comparison'.format(ds_lp_fp.sat_id), fontsize='medium', append_time=True)

    db.save_figure(file_dir=file_dir_figure, file_name='example_EFI_LP_FP_vs_EFI_LP_1B_Swarm-{}_comparison'.format(ds_lp_fp.sat_id), dpi=100, append_time=False)
    db.show()

if __name__ == "__main__":
    # test_swarm_EFI_LP_1B_overview()
    # test_swarm_EFI_LPI_1B_overview()
    #test_swarm_EFI_LP_FP_overview()
    test_swarm_compare_EFI_LP_FP()