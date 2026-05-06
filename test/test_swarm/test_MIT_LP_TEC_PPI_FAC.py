import datetime
import matplotlib.pyplot as plt
import numpy as np
import pathlib

import geospacelab.visualization.mpl.dashboards as dashboards

cwd = pathlib.Path(__file__).parent.resolve()
file_dir_figure = cwd / 'figures'
file_dir_figure.mkdir(parents=True, exist_ok=True)

def test_swarm_MIT_LP_overview_1():
    """Test Swarm MIT LP data product
    
    """
    dt_fr = datetime.datetime(2016, 3, 14, 0, 0)
    dt_to = datetime.datetime(2016, 3, 15, 23, 59)

    db = dashboards.TSDashboard(
        dt_fr=dt_fr, dt_to=dt_to, figure_config={'figsize': (8, 12)},
        )

    ds = db.dock(datasource_contents=['esa_eo', 'swarm', 'l2daily', 'mit_lp'], sat_id='A', add_APEX=True,)

    panel_layouts = [
        [ds['n_e']],
        [ds['T_e']],
        [ds['Depth']],
        [ds['DR']],
        [ds['Width']],
        [ds['dL']],
        [ds['PW_Gradient'], ds['EW_Gradient']],
        [ds['QUALITY']],
    ]

    db.set_layout(panel_layouts=panel_layouts)
    db.draw()
    db.add_title(title='Swarm-{} MIT LP Overview'.format(ds.sat_id), fontsize='medium', append_time=True)

    db.save_figure(file_dir=file_dir_figure, file_name='example_MIT_LP_Swarm-{}_overview'.format(ds.sat_id), dpi=100, append_time=False)
    db.show()
    
def test_swarm_MIT_LP_overview_2():
    """Test Swarm MIT LP data product
    
    """
    dt_fr = datetime.datetime(2016, 3, 14, 0, 0)
    dt_to = datetime.datetime(2016, 3, 15, 23, 59)

    db = dashboards.TSDashboard(
        dt_fr=dt_fr, dt_to=dt_to, figure_config={'figsize': (8, 12)},
        )

    ds = db.dock(datasource_contents=['esa_eo', 'swarm', 'l2daily', 'mit_lp'], sat_id='A', add_APEX=True,)

    panel_layouts = [
        [ds['n_e_ID']],
        [ds['T_e_ID']],
        [ds['GEO_LAT_ID']],
        [ds['GEO_LON_ID']],
        [ds['Position_Quality_ID']],
    ]

    db.set_layout(panel_layouts=panel_layouts)
    db.draw()
    db.add_title(title='Swarm-{} MIT LP Overview'.format(ds.sat_id), fontsize='medium', append_time=True)

    db.save_figure(file_dir=file_dir_figure, file_name='example_MIT_LP_Swarm-{}_overview_2'.format(ds.sat_id), dpi=100, append_time=False)
    db.show()
   
    
def test_swarm_MIT_TEC_overview():
    """Test Swarm MIT TEC data product
    
    """
    dt_fr = datetime.datetime(2016, 3, 14, 0, 0)
    dt_to = datetime.datetime(2016, 3, 15, 23, 59)

    db = dashboards.TSDashboard(
        dt_fr=dt_fr, dt_to=dt_to, figure_config={'figsize': (8, 12)},
        )

    ds = db.dock(datasource_contents=['esa_eo', 'swarm', 'l2daily', 'mit_tec'], sat_id='A', add_APEX=True,)

    panel_layouts = [
        [ds['TEC']],
        [ds['Depth']],
        [ds['DR']],
        [ds['Width']],
        [ds['dL']],
        [ds['PW_Gradient'], ds['EW_Gradient']],
        [ds['QUALITY']],
        [ds['TEC_ID']],
        [ds['GEO_LAT_ID']],
        [ds['Position_Quality_ID']],
    ]

    db.set_layout(panel_layouts=panel_layouts)
    db.draw()
    db.add_title(title='Swarm-{} MIT TEC Overview'.format(ds.sat_id), fontsize='medium', append_time=True)

    db.save_figure(file_dir=file_dir_figure, file_name='example_MIT_TEC_Swarm-{}_overview'.format(ds.sat_id), dpi=100, append_time=False)
    db.show()
    
    
def test_swarm_PPI_FAC_overview():
    """Test Swarm PPI FAC data product
    
    """
    dt_fr = datetime.datetime(2016, 3, 14, 0, 0)
    dt_to = datetime.datetime(2016, 3, 15, 23, 59)

    db = dashboards.TSDashboard(
        dt_fr=dt_fr, dt_to=dt_to, figure_config={'figsize': (8, 12)},
        )

    ds = db.dock(datasource_contents=['esa_eo', 'swarm', 'l2daily', 'ppi_fac'], sat_id='A', add_APEX=True,)

    panel_layouts = [
        [ds['PPI']],
        [ds['Sigma']],
        [ds['QUALITY']],
    ]

    db.set_layout(panel_layouts=panel_layouts)
    db.draw()
    db.add_title(title='Swarm-{} PPI FAC Overview'.format(ds.sat_id), fontsize='medium', append_time=True)

    db.save_figure(file_dir=file_dir_figure, file_name='example_PPI_FAC_Swarm-{}_overview'.format(ds.sat_id), dpi=100, append_time=False)
    db.show()
   

if __name__ == "__main__":
    # test_swarm_MIT_LP_overview_1()
    # test_swarm_MIT_LP_overview_2()
    
    # test_swarm_MIT_TEC_overview()
    test_swarm_PPI_FAC_overview()