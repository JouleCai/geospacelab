import datetime
import matplotlib.pyplot as plt
import numpy as np
import pathlib

import geospacelab.visualization.mpl.dashboards as dashboards

cwd = pathlib.Path(__file__).parent.resolve()
file_dir_figure = cwd / 'figures'
file_dir_figure.mkdir(parents=True, exist_ok=True)

def test_swarm_IBI_TMS_overview():
    """Test Swarm IBI TMS data product
    
    """
    dt_fr = datetime.datetime(2016, 3, 14, 0, 0)
    dt_to = datetime.datetime(2016, 3, 15, 23, 59)

    db = dashboards.TSDashboard(
        dt_fr=dt_fr, dt_to=dt_to, figure_config={'figsize': (8, 12)},
        )

    ds = db.dock(datasource_contents=['esa_eo', 'swarm', 'l2daily', 'ibi_tms'], sat_id='A', add_APEX=True,)

    panel_layouts = [
        [ds['Bubble_Index']],
        [ds['Bubble_Probability']],
        [ds['FLAG'], ds['FLAG_F'], ds['FLAG_B'], ds['FLAG_q']],
        [ds['FLAG_BIN_AUX']],
        [ds['FLAG_F_BIN_AUX']], 
        [ds['FLAG_B_BIN_AUX']], 
        [ds['FLAG_q_BIN_AUX']]
    ]

    db.set_layout(panel_layouts=panel_layouts)
    db.draw()
    db.add_title(title='Swarm-{} IBI TMS Overview'.format(ds.sat_id), fontsize='medium', append_time=True)

    db.save_figure(file_dir=file_dir_figure, file_name='example_IBI_TMS_Swarm-{}_overview'.format(ds.sat_id), dpi=100, append_time=False)
    db.show()
    
def test_swarm_IBI_TMS_zoom():
    """Test Swarm IBI TMS data product
    
    """
    dt_fr = datetime.datetime(2016, 3, 15, 18, 00)
    dt_to = datetime.datetime(2016, 3, 15, 19, 30)

    db = dashboards.TSDashboard(
        dt_fr=dt_fr, dt_to=dt_to, figure_config={'figsize': (8, 12)},
        timeline_extra_labels=['GEO_LAT', 'GEO_LON', 'APEX_LAT', 'APEX_LON', 'APEX_MLT',]  # Not applicable for very scattered data points like AEJ_PBS peaks
        )

    ds = db.dock(datasource_contents=['esa_eo', 'swarm', 'l2daily', 'ibi_tms'], sat_id='A', add_APEX=True,)

    panel_layouts = [
        [ds['Bubble_Index']],
        [ds['Bubble_Probability']],
        [ds['FLAG'], ds['FLAG_F'], ds['FLAG_B'], ds['FLAG_q']],
        [ds['FLAG_BIN_AUX']],
        [ds['FLAG_F_BIN_AUX']], 
        [ds['FLAG_B_BIN_AUX']], 
        [ds['FLAG_q_BIN_AUX']]
    ]

    db.set_layout(panel_layouts=panel_layouts)
    db.draw()
    db.add_title(title='Swarm-{} IBI TMS Zoom'.format(ds.sat_id), fontsize='medium', append_time=True)
    
    db.save_figure(file_dir=file_dir_figure, file_name='example_IBI_TMS_Swarm-{}_zoom'.format(ds.sat_id), dpi=100, append_time=False)
    db.show()
    


if __name__ == "__main__":
    # test_swarm_IBI_TMS_overview()
    test_swarm_IBI_TMS_zoom()
