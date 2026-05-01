import datetime
import matplotlib.pyplot as plt
import numpy as np
import pathlib

import geospacelab.visualization.mpl.dashboards as dashboards

cwd = pathlib.Path(__file__).parent.resolve()
file_dir_figure = cwd / 'figures'
file_dir_figure.mkdir(parents=True, exist_ok=True)

def test_swarm_AEJ_LPL_overview():
    """Test Swarm AEJ/LPL data product
    
    """
    dt_fr = datetime.datetime(2024, 5, 10, 12, 0)
    dt_to = datetime.datetime(2024, 5, 11, 12, 0)

    db = dashboards.TSDashboard(
        dt_fr=dt_fr, dt_to=dt_to, figure_config={'figsize': (8, 8)},
        )

    ds = db.dock(datasource_contents=['esa_eo', 'swarm', 'l2daily', 'aej_lpl'], sat_id='C', add_APEX=True, add_AACGM=True)

    glat = ds['GEO_LAT']
    glon = ds['GEO_LON']
    J_N = ds['J_N']
    J_E = ds['J_E']
    J_QD = ds['J_QD']
    
    panel_layouts = [
        [J_N, J_E],
        [J_QD],
        [glat, [glon]]
    ]

    db.set_layout(panel_layouts=panel_layouts)
    db.draw()
    db.add_title(title='Swarm AEJ/LPL Overview', fontsize='medium', append_time=True)

    db.save_figure(file_dir=file_dir_figure, file_name='example_AEJ_LPL_Swarm-{}_overview'.format(ds.sat_id), dpi=100, append_time=False)
    db.show()
    
def test_swarm_AEJ_LPL_zoom():
    """Test Swarm AEJ/LPL data product
    
    """
    dt_fr = datetime.datetime(2024, 5, 10, 20, 30)
    dt_to = datetime.datetime(2024, 5, 10, 21, 0)

    db = dashboards.TSDashboard(
        dt_fr=dt_fr, dt_to=dt_to, figure_config={'figsize': (8, 8)},
        timeline_extra_labels=['GEO_LAT', 'GEO_LON', 'APEX_LAT', 'APEX_LON', 'APEX_MLT',] 
        )

    ds = db.dock(datasource_contents=['esa_eo', 'swarm', 'l2daily', 'aej_lpl'], sat_id='C', add_APEX=True, add_AACGM=True)

    glat = ds['GEO_LAT']
    glon = ds['GEO_LON']
    J_N = ds['J_N']
    J_E = ds['J_E']
    J_QD = ds['J_QD']
    
    panel_layouts = [
        [J_N, J_E],
        [J_QD],
        [glat, [glon]]
    ]

    db.set_layout(panel_layouts=panel_layouts)
    db.draw()
    db.add_title(title='Swarm AEJ/LPL Zoom', fontsize='medium', append_time=True)
    
    db.save_figure(file_dir=file_dir_figure, file_name='example_AEJ_LPL_Swarm-{}_zoom'.format(ds.sat_id), dpi=100, append_time=False)
    db.show()


if __name__ == "__main__":
    test_swarm_AEJ_LPL_overview()
    test_swarm_AEJ_LPL_zoom()
