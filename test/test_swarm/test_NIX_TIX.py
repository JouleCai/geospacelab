import datetime
import matplotlib.pyplot as plt
import numpy as np
import pathlib

import geospacelab.visualization.mpl.dashboards as dashboards

cwd = pathlib.Path(__file__).parent.resolve()
file_dir_figure = cwd / 'figures'
file_dir_figure.mkdir(parents=True, exist_ok=True)

def test_swarm_NIX_TMS_overview():
    """Test Swarm NIX TMS data product
    
    """
    dt_fr = datetime.datetime(2016, 3, 14, 0, 0)
    dt_to = datetime.datetime(2016, 3, 15, 23, 59)

    db = dashboards.TSDashboard(
        dt_fr=dt_fr, dt_to=dt_to, figure_config={'figsize': (8, 12)},
        )

    ds = db.dock(datasource_contents=['esa_eo', 'swarm', 'l2daily', 'nix_tms'],)  # Only for "AC"

    panel_layouts = [
        [ds['Negix_X']],
        [ds['Negix_Y']],
        [ds['Negix_Total']],
        [ds['N_Measurements']],
        [ds['Flag_Negix']],
    ]

    db.set_layout(panel_layouts=panel_layouts)
    db.draw()
    db.add_title(title='Swarm-{} NIX TMS Overview'.format(ds.sat_id), fontsize='medium', append_time=True)

    db.save_figure(file_dir=file_dir_figure, file_name='example_NIX_TMS_Swarm-{}_overview'.format(ds.sat_id), dpi=100, append_time=False)
    db.show()
    
    
def test_swarm_NIX_TMS_zoom():
    """Test Swarm NIX TMS data product
    
    """
    dt_fr = datetime.datetime(2016, 3, 14, 20, 0)
    dt_to = datetime.datetime(2016, 3, 14, 20, 30)

    db = dashboards.TSDashboard(
        dt_fr=dt_fr, dt_to=dt_to, figure_config={'figsize': (8, 12)},
        )

    ds = db.dock(datasource_contents=['esa_eo', 'swarm', 'l2daily', 'nix_tms'],)

    panel_layouts = [
        [ds['Negix_X']],
        [ds['Negix_Y']],
        [ds['Negix_Total']],
        [ds['N_Measurements']],
        [ds['Flag_Negix']],
    ]

    db.set_layout(panel_layouts=panel_layouts)
    db.draw()
    db.add_title(title='Swarm-{} NIX TMS Overview'.format(ds.sat_id), fontsize='medium', append_time=True)

    db.save_figure(file_dir=file_dir_figure, file_name='example_NIX_TMS_Swarm-{}_overview'.format(ds.sat_id), dpi=100, append_time=False)
    db.show()
    
def test_swarm_TIX_TMS_zoom():
    """Test Swarm TIX TMS data product
    
    """
    dt_fr = datetime.datetime(2016, 3, 14, 20, 0)
    dt_to = datetime.datetime(2016, 3, 14, 20, 30)

    db = dashboards.TSDashboard(
        dt_fr=dt_fr, dt_to=dt_to, figure_config={'figsize': (8, 12)},
        )

    ds = db.dock(datasource_contents=['esa_eo', 'swarm', 'l2daily', 'tix_tms'],)

    panel_layouts = [
        [ds['Tegix_X']],
        [ds['Tegix_Y']],
        [ds['Tegix_Total']],
        [ds['N_Measurements']],
        [ds['Flag_Tegix']],
    ]

    db.set_layout(panel_layouts=panel_layouts)
    db.draw()
    db.add_title(title='Swarm-{} TIX TMS Overview'.format(ds.sat_id), fontsize='medium', append_time=True)

    db.save_figure(file_dir=file_dir_figure, file_name='example_TIX_TMS_Swarm-{}_overview'.format(ds.sat_id), dpi=100, append_time=False)
    db.show()
   

if __name__ == "__main__":
    # test_swarm_NIX_TMS_overview()
    # test_swarm_NIX_TMS_zoom()
    test_swarm_TIX_TMS_zoom()