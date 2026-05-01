import datetime
import matplotlib.pyplot as plt
import numpy as np
import pathlib

import geospacelab.visualization.mpl.dashboards as dashboards

cwd = pathlib.Path(__file__).parent.resolve()
file_dir_figure = cwd / 'figures'
file_dir_figure.mkdir(parents=True, exist_ok=True)

def test_swarm_AEJ_PBS_overview():
    """Test Swarm AEJ/PBS data product
    
    """
    dt_fr = datetime.datetime(2024, 5, 10, 12, 0)
    dt_to = datetime.datetime(2024, 5, 11, 12, 0)

    db = dashboards.TSDashboard(
        dt_fr=dt_fr, dt_to=dt_to, figure_config={'figsize': (8, 8)},
        )

    ds = db.dock(datasource_contents=['esa_eo', 'swarm', 'l2daily', 'aej_pbs'], sat_id='C', add_APEX=True, add_AACGM=True)

    panel_layouts = [
        [ds['WEJ_PEAK'], ds['EEJ_PEAK']],
        [
            ds['GEO_LAT_WEJ_PB'],
            ds['GEO_LAT_WEJ_PEAK'], 
            ds['GEO_LAT_WEJ_EB'],
            ds['GEO_LAT_EEJ_PB'],
            ds['GEO_LAT_EEJ_PEAK'],
            ds['GEO_LAT_EEJ_EB'],
        ],
        [
            ds['QD_LAT_WEJ_PB'],
            ds['QD_LAT_WEJ_PEAK'],
            ds['QD_LAT_WEJ_EB'],
            ds['QD_LAT_EEJ_PB'],
            ds['QD_LAT_EEJ_PEAK'],
            ds['QD_LAT_EEJ_EB'],
        ],
        [
            ds['QUALITY_FLAG'],
        ],
    ]

    db.set_layout(panel_layouts=panel_layouts)
    db.draw()
    db.add_title(title='Swarm AEJ/PBS Overview', fontsize='medium', append_time=True)

    db.save_figure(file_dir=file_dir_figure, file_name='example_AEJ_PBS_Swarm-{}_overview'.format(ds.sat_id), dpi=100, append_time=False)
    db.show()
    
def test_swarm_AEJ_PBS_zoom():
    """Test Swarm AEJ/PBS data product
    
    """
    dt_fr = datetime.datetime(2024, 5, 10, 20, 30)
    dt_to = datetime.datetime(2024, 5, 10, 21, 0)

    db = dashboards.TSDashboard(
        dt_fr=dt_fr, dt_to=dt_to, figure_config={'figsize': (8, 8)},
        # timeline_extra_labels=['GEO_LAT', 'GEO_LON', 'APEX_LAT', 'APEX_LON', 'APEX_MLT',]  # Not applicable for very scattered data points like AEJ_PBS peaks
        )

    ds = db.dock(datasource_contents=['esa_eo', 'swarm', 'l2daily', 'aej_pbs'], sat_id='C', add_APEX=True, add_AACGM=True)

    panel_layouts = [
        [ds['WEJ_PEAK'], ds['EEJ_PEAK']],
        [
            ds['GEO_LAT_WEJ_PB'],
            ds['GEO_LAT_WEJ_PEAK'], 
            ds['GEO_LAT_WEJ_EB'],
            ds['GEO_LAT_EEJ_PB'],
            ds['GEO_LAT_EEJ_PEAK'],
            ds['GEO_LAT_EEJ_EB'],
            ],
        [
            ds['QD_LAT_WEJ_PB'],
            ds['QD_LAT_WEJ_PEAK'],
            ds['QD_LAT_WEJ_EB'],
            ds['QD_LAT_EEJ_PB'],
            ds['QD_LAT_EEJ_PEAK'],
            ds['QD_LAT_EEJ_EB'],
            ],
        [
            ds['QUALITY_FLAG'],
        ],
    ]

    db.set_layout(panel_layouts=panel_layouts)
    db.draw()
    db.add_title(title='Swarm AEJ/PBS Zoom', fontsize='medium', append_time=True)
    
    db.save_figure(file_dir=file_dir_figure, file_name='example_AEJ_PBS_Swarm-{}_zoom'.format(ds.sat_id), dpi=100, append_time=False)
    db.show()


if __name__ == "__main__":
    test_swarm_AEJ_PBS_overview()
    test_swarm_AEJ_PBS_zoom()
