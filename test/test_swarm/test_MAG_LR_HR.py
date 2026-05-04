import datetime
import matplotlib.pyplot as plt
import numpy as np
import pathlib

import geospacelab.visualization.mpl.dashboards as dashboards

cwd = pathlib.Path(__file__).parent.resolve()
file_dir_figure = cwd / 'figures'
file_dir_figure.mkdir(parents=True, exist_ok=True)

def test_swarm_MAG_LR_overview():
    """Test Swarm MAG LR data product
    
    """
    dt_fr = datetime.datetime(2016, 3, 14, 12, 0)
    dt_to = datetime.datetime(2016, 3, 15, 23, 59)

    db = dashboards.TSDashboard(
        dt_fr=dt_fr, dt_to=dt_to, figure_config={'figsize': (8, 12)},
        )

    ds = db.dock(datasource_contents=['esa_eo', 'swarm', 'l1b', 'mag_lr'], sat_id='A', add_APEX=True,)

    panel_layouts = [
        [ds['F']],
        [ds['B_N'], ds['B_E'], ds['B_C']],
        [ds['B_VFM_x'], ds['B_VFM_y'], ds['B_VFM_z']],
        [ds['dB_Sun_VFM_x'], ds['dB_Sun_VFM_y'], ds['dB_Sun_VFM_z']],
        [ds['dB_AOCS_VFM_x'], ds['dB_AOCS_VFM_y'], ds['dB_AOCS_VFM_z']],
        [ds['dB_other_VFM_x'], ds['dB_other_VFM_y'], ds['dB_other_VFM_z']],
        [ds['FLAG_F_BIN_AUX']],
        [ds['FLAG_B_BIN_AUX']],
        [ds['FLAG_q_BIN_AUX']],
        [ds['FLAG_Platform_BIN_AUX']],
    ]

    db.set_layout(panel_layouts=panel_layouts)
    db.draw()
    db.add_title(title='Swarm-{} MAG LR Overview'.format(ds.sat_id), fontsize='medium', append_time=True)

    db.save_figure(file_dir=file_dir_figure, file_name='example_MAG_LR_Swarm-{}_overview'.format(ds.sat_id), dpi=100, append_time=False)
    db.show()
    

def test_swarm_MAG_LR_zoom():
    """Test Swarm MAG LR data product
    
    """
    dt_fr = datetime.datetime(2016, 3, 15, 18, 16)
    dt_to = datetime.datetime(2016, 3, 15, 18, 23)

    db = dashboards.TSDashboard(
        dt_fr=dt_fr, dt_to=dt_to, figure_config={'figsize': (8, 12)},
        timeline_extra_labels=['GEO_LAT', 'GEO_LON', 'APEX_LAT', 'APEX_LON', 'APEX_MLT',]  # Not applicable for very scattered data points like AEJ_PBS peaks
        )

    ds = db.dock(datasource_contents=['esa_eo', 'swarm', 'l1b', 'mag_lr'], sat_id='A', add_APEX=True,)

    panel_layouts = [
        [ds['F']],
        [ds['B_N'], ds['B_E'], ds['B_C']],
        [ds['B_VFM_x'], ds['B_VFM_y'], ds['B_VFM_z']],
        [ds['dB_Sun_VFM_x'], ds['dB_Sun_VFM_y'], ds['dB_Sun_VFM_z']],
        [ds['dB_AOCS_VFM_x'], ds['dB_AOCS_VFM_y'], ds['dB_AOCS_VFM_z']],
        [ds['dB_other_VFM_x'], ds['dB_other_VFM_y'], ds['dB_other_VFM_z']],
        [ds['FLAG_F_BIN_AUX']],
        [ds['FLAG_B_BIN_AUX']],
        [ds['FLAG_q_BIN_AUX']],
        [ds['FLAG_Platform_BIN_AUX']],
    ]

    db.set_layout(panel_layouts=panel_layouts)
    db.draw()
    db.add_title(title='Swarm-{} MAG LR Zoom'.format(ds.sat_id), fontsize='medium', append_time=True)
    
    db.save_figure(file_dir=file_dir_figure, file_name='example_MAG_LR_Swarm-{}_zoom'.format(ds.sat_id), dpi=100, append_time=False)
    db.show()

    
def test_swarm_MAG_HR_zoom():
    """Test Swarm MAG HR data product
    
    """
    dt_fr = datetime.datetime(2016, 3, 15, 18, 16)
    dt_to = datetime.datetime(2016, 3, 15, 18, 23)

    db = dashboards.TSDashboard(
        dt_fr=dt_fr, dt_to=dt_to, figure_config={'figsize': (8, 12)},
        timeline_extra_labels=['GEO_LAT', 'GEO_LON', 'APEX_LAT', 'APEX_LON', 'APEX_MLT',]  # Not applicable for very scattered data points like AEJ_PBS peaks
        )

    ds = db.dock(datasource_contents=['esa_eo', 'swarm', 'l1b', 'mag_hr'], sat_id='A', add_APEX=True,)

    panel_layouts = [
        [ds['B_N'], ds['B_E'], ds['B_C']],
        [ds['B_VFM_x'], ds['B_VFM_y'], ds['B_VFM_z']],
        [ds['dB_Sun_VFM_x'], ds['dB_Sun_VFM_y'], ds['dB_Sun_VFM_z']],
        [ds['dB_AOCS_VFM_x'], ds['dB_AOCS_VFM_y'], ds['dB_AOCS_VFM_z']],
        [ds['dB_other_VFM_x'], ds['dB_other_VFM_y'], ds['dB_other_VFM_z']],
        [ds['FLAG_B_BIN_AUX']],
        [ds['FLAG_q_BIN_AUX']],
        [ds['FLAG_Platform_BIN_AUX']],
    ]

    db.set_layout(panel_layouts=panel_layouts)
    db.draw()
    db.add_title(title='Swarm-{} MAG HR Zoom'.format(ds.sat_id), fontsize='medium', append_time=True)
    
    db.save_figure(file_dir=file_dir_figure, file_name='example_MAG_HR_Swarm-{}_zoom'.format(ds.sat_id), dpi=100, append_time=False)
    db.show()
    


if __name__ == "__main__":
    # test_swarm_MAG_LR_overview()
    test_swarm_MAG_LR_zoom()
    #test_swarm_MAG_HR_zoom()
