import datetime
import matplotlib.pyplot as plt
import numpy as np
import pathlib

import geospacelab.visualization.mpl.dashboards as dashboards

cwd = pathlib.Path(__file__).parent.resolve()
file_dir_figure = cwd / 'figures'
file_dir_figure.mkdir(parents=True, exist_ok=True)

def test_swarm_EFI_IDM_overview():
    """Test Swarm EFI IDM data product
    
    """
    dt_fr = datetime.datetime(2016, 3, 14, 15, 0)
    dt_to = datetime.datetime(2016, 3, 14, 22, 0)

    db = dashboards.TSDashboard(
        dt_fr=dt_fr, dt_to=dt_to, figure_config={'figsize': (8, 12)},
        )

    ds = db.dock(datasource_contents=['esa_eo', 'swarm', 'advanced', 'efi_idm'], sat_id='A', add_APEX=True,)

    panel_layouts = [
        [ds['M_i_eff_model'], ds['M_i_eff']],
        [ds['FLAG_M_i_eff_BIN_AUX']],
        [ds['v_i_raw'], ds['v_i']],
        [ds['FLAG_v_i_BIN_AUX']],
        [ds['n_i'],],
        [ds['FLAG_n_i_BIN_AUX']],
    ]

    db.set_layout(panel_layouts=panel_layouts)
    db.draw()
    db.add_title(title='Swarm-{} EFI IDM Overview'.format(ds.sat_id), fontsize='medium', append_time=True)

    db.save_figure(file_dir=file_dir_figure, file_name='example_EFI_IDM_Swarm-{}_overview'.format(ds.sat_id), dpi=100, append_time=False)
    db.show()
    
def test_swarm_EFI_IDM_zoom():
    """Test Swarm EFI IDM data product
    
    """
    dt_fr = datetime.datetime(2016, 3, 14, 20, 15)
    dt_to = datetime.datetime(2016, 3, 14, 21, 0)

    db = dashboards.TSDashboard(
        dt_fr=dt_fr, dt_to=dt_to, figure_config={'figsize': (8, 12)},
        timeline_extra_labels=['GEO_LAT', 'GEO_LON', 'APEX_LAT', 'APEX_LON', 'APEX_MLT',]  # Not applicable for very scattered data points like AEJ_PBS peaks
        )

    ds = db.dock(datasource_contents=['esa_eo', 'swarm', 'advanced', 'efi_idm'], sat_id='A', add_APEX=True,)

    panel_layouts = [
        [ds['M_i_eff_model'], ds['M_i_eff']],
        [ds['FLAG_M_i_eff_BIN_AUX']],
        [ds['v_i_raw'], ds['v_i']],
        [ds['FLAG_v_i_BIN_AUX']],
        [ds['n_i'],],
        [ds['FLAG_n_i_BIN_AUX']],
    ]

    db.set_layout(panel_layouts=panel_layouts)
    db.draw()
    db.add_title(title='Swarm-{} EFI IDM Zoom'.format(ds.sat_id), fontsize='medium', append_time=True)
    
    db.save_figure(file_dir=file_dir_figure, file_name='example_EFI_IDM_Swarm-{}_zoom'.format(ds.sat_id), dpi=100, append_time=False)
    db.show()
    


if __name__ == "__main__":
    test_swarm_EFI_IDM_overview()
    test_swarm_EFI_IDM_zoom()
