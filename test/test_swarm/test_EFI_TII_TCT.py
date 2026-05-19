import datetime
import matplotlib.pyplot as plt
import numpy as np
import pathlib

import geospacelab.visualization.mpl.dashboards as dashboards

cwd = pathlib.Path(__file__).parent.resolve()
file_dir_figure = cwd / 'figures'
file_dir_figure.mkdir(parents=True, exist_ok=True)

def test_swarm_EFI_TII_TCT02_overview():
    """Test Swarm EFI TII TCT02 data product
    
    """
    dt_fr = datetime.datetime(2015, 3, 17, 8, 0)
    dt_to = datetime.datetime(2015, 3, 17, 18, 0)

    db = dashboards.TSDashboard(
        dt_fr=dt_fr, dt_to=dt_to, figure_config={'figsize': (8, 12)},
        )

    ds = db.dock(datasource_contents=['esa_eo', 'swarm', 'advanced', 'efi_tct02'], sat_id='A', add_APEX=True,)

    panel_layouts = [
        [ds['v_i_H_x'], ds['v_i_V_x'], ds['v_i_H_y'], ds['v_i_V_z']],
        [ds['v_SC_N'], ds['v_SC_E'], ds['v_SC_C']],
        [ds['E_H_x'], ds['E_H_y'], ds['E_H_z']],
        [ds['E_V_x'], ds['E_V_y'], ds['E_V_z']],
        [ds['B_x'], ds['B_y'], ds['B_z']],
        [ds['v_i_CR_x'], ds['v_i_CR_y'], ds['v_i_CR_z']],
        [ds['QUALITY_FLAG_BIN_AUX'],],
        [ds['CALIB_FLAG_BIN_AUX'],],
    ]

    db.set_layout(panel_layouts=panel_layouts)
    db.draw()
    db.add_title(title='Swarm-{} EFI TII TCT02 Overview'.format(ds.sat_id), fontsize='medium', append_time=True)

    db.save_figure(file_dir=file_dir_figure, file_name='example_EFI_TII_TCT02_Swarm-{}_overview'.format(ds.sat_id), dpi=100, append_time=False)
    db.show()
    
def test_swarm_EFI_TII_TCT02_zoom():
    """Test Swarm EFI TII TCT02 data product
    
    """
    dt_fr = datetime.datetime(2015, 3, 17, 12, 40)
    dt_to = datetime.datetime(2015, 3, 17, 13, 10)

    db = dashboards.TSDashboard(
        dt_fr=dt_fr, dt_to=dt_to, figure_config={'figsize': (8, 12)},
        timeline_extra_labels=['GEO_LAT', 'GEO_LON', 'APEX_LAT', 'APEX_LON', 'APEX_MLT']  # Not applicable for very scattered data points like AEJ_PBS peaks
        )

    ds = db.dock(datasource_contents=['esa_eo', 'swarm', 'advanced', 'efi_tct02'], sat_id='A', add_APEX=True,)

    panel_layouts = [
        [ds['v_i_H_x'], ds['v_i_V_x'], ds['v_i_H_y'], ds['v_i_V_z']],
        [ds['v_SC_N'], ds['v_SC_E'], ds['v_SC_C']],
        [ds['E_H_x'], ds['E_H_y'], ds['E_H_z']],
        [ds['E_V_x'], ds['E_V_y'], ds['E_V_z']],
        [ds['B_x'], ds['B_y'], ds['B_z']],
        [ds['v_i_CR_x'], ds['v_i_CR_y'], ds['v_i_CR_z']],
        [ds['QUALITY_FLAG_BIN_AUX'],],
        [ds['CALIB_FLAG_BIN_AUX'],],
    ]

    db.set_layout(panel_layouts=panel_layouts)
    db.draw()
    db.add_title(title='Swarm-{} EFI TII TCT02 Zoom'.format(ds.sat_id), fontsize='medium', append_time=True)
    
    db.save_figure(file_dir=file_dir_figure, file_name='example_EFI_TII_TCT02_Swarm-{}_zoom'.format(ds.sat_id), dpi=100, append_time=False)
    db.show()
    

def test_swarm_EFI_TII_TCT16_zoom():
    """Test Swarm EFI TII TCT16 data product
    
    """
    dt_fr = datetime.datetime(2015, 3, 17, 12, 40)
    dt_to = datetime.datetime(2015, 3, 17, 13, 10)

    db = dashboards.TSDashboard(
        dt_fr=dt_fr, dt_to=dt_to, figure_config={'figsize': (8, 12)},
        )

    ds = db.dock(datasource_contents=['esa_eo', 'swarm', 'advanced', 'efi_tct16'], sat_id='A', add_APEX=True,)

    panel_layouts = [
        [ds['v_i_H_x'], ds['v_i_V_x'], ds['v_i_H_y'], ds['v_i_V_z']],
        [ds['v_SC_N'], ds['v_SC_E'], ds['v_SC_C']],
        [ds['E_H_x'], ds['E_H_y'], ds['E_H_z']],
        [ds['E_V_x'], ds['E_V_y'], ds['E_V_z']],
        [ds['B_x'], ds['B_y'], ds['B_z']],
        [ds['v_i_CR_x'], ds['v_i_CR_y'], ds['v_i_CR_z']],
        [ds['QUALITY_FLAG_BIN_AUX'],],
        [ds['CALIB_FLAG_BIN_AUX'],],
    ]

    db.set_layout(panel_layouts=panel_layouts)
    db.draw()
    db.add_title(title='Swarm-{} EFI TII TCT16 Zoom'.format(ds.sat_id), fontsize='medium', append_time=True)

    db.save_figure(file_dir=file_dir_figure, file_name='example_EFI_TII_TCT16_Swarm-{}_zoom'.format(ds.sat_id), dpi=100, append_time=False)
    db.show()


if __name__ == "__main__":
    # test_swarm_EFI_TII_TCT02_overview()
    test_swarm_EFI_TII_TCT02_zoom()
    # test_swarm_EFI_TII_TCT16_zoom()
