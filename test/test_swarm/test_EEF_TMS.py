import datetime
import matplotlib.pyplot as plt
import numpy as np
import pathlib

import geospacelab.visualization.mpl.dashboards as dashboards

cwd = pathlib.Path(__file__).parent.resolve()
file_dir_figure = cwd / 'figures'
file_dir_figure.mkdir(parents=True, exist_ok=True)

def test_swarm_EEF_TMS_overview():
    """Test Swarm EEF/TMS data product
    
    """
    dt_fr = datetime.datetime(2024, 5, 10, 12, 0)
    dt_to = datetime.datetime(2024, 5, 11, 12, 0)

    db = dashboards.TSDashboard(
        dt_fr=dt_fr, dt_to=dt_to, figure_config={'figsize': (8, 8)},
        )

    ds = db.dock(
        datasource_contents=['esa_eo', 'swarm', 'l2daily', 'eef_tms'], 
        product_version='latest', # 'latest' (default), '0301', 
        sat_id='C', )
    
    panel_layouts = [
        [ds['EF_EQ']],
        [ds['EEJ_E']],
        [ds['EEJ_N']], # After version 0502, EEJ_E and EEJ_N are provided separately. Before that, only EEJ_E
        [ds['Relative_Error']],
        [ds['FLAG']],
    ]

    db.set_layout(panel_layouts=panel_layouts)
    db.draw()
    db.add_title(title='Swarm-{} EEF/TMS Overview'.format(ds.sat_id), fontsize='medium', append_time=True)

    db.save_figure(file_dir=file_dir_figure, file_name='example_EEF_TMS_Swarm-{}_overview'.format(ds.sat_id), dpi=100, append_time=False)
    db.show()


if __name__ == "__main__":
    test_swarm_EEF_TMS_overview()
