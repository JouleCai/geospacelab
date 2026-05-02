import datetime
import matplotlib.pyplot as plt
import numpy as np
import pathlib

import geospacelab.visualization.mpl.dashboards as dashboards

cwd = pathlib.Path(__file__).parent.resolve()
file_dir_figure = cwd / 'figures'
file_dir_figure.mkdir(parents=True, exist_ok=True)

def test_swarm_AOB_FAC_overview():
    """Test Swarm AOB/FAC data product
    
    """
    dt_fr = datetime.datetime(2024, 5, 10, 12, 0)
    dt_to = datetime.datetime(2024, 5, 11, 12, 0)

    db = dashboards.TSDashboard(
        dt_fr=dt_fr, dt_to=dt_to, figure_config={'figsize': (8, 8)},
        )

    ds = db.dock(datasource_contents=['esa_eo', 'swarm', 'l2daily', 'aob_fac'], sat_id='C', add_APEX=True, add_AACGM=True)
    
    panel_layouts = [
        [ds['SC_GEO_LAT']],
        [ds['BOUNDARY_FLAG']],
        [ds['SC_GEO_LAT_EB'], ds['SC_GEO_LAT_PB']],
        [ds['SC_QD_LAT_EB'], ds['SC_QD_LAT_PB']],
        [ds['QUALITY_Pa_EB'], ds['QUALITY_Pa_PB']],
        [ds['QUALITY_Sigma_EB'], ds['QUALITY_Sigma_PB']],
    ]

    db.set_layout(panel_layouts=panel_layouts)
    db.draw()
    db.add_title(title='Swarm-{} AOB/FAC Overview'.format(ds.sat_id), fontsize='medium', append_time=True)

    db.save_figure(file_dir=file_dir_figure, file_name='example_AOB_FAC_Swarm-{}_overview'.format(ds.sat_id), dpi=100, append_time=False)
    db.show()
    
def test_swarm_AOB_FAC_zoom():
    """Test Swarm AOB/FAC data product
    
    """
    dt_fr = datetime.datetime(2024, 5, 10, 20, 30)
    dt_to = datetime.datetime(2024, 5, 10, 21, 0)

    db = dashboards.TSDashboard(
        dt_fr=dt_fr, dt_to=dt_to, figure_config={'figsize': (8, 8)},
        # timeline_extra_labels=['GEO_LAT', 'GEO_LON', 'APEX_LAT', 'APEX_LON', 'APEX_MLT',] 
        )

    ds = db.dock(datasource_contents=['esa_eo', 'swarm', 'l2daily', 'aob_fac'], sat_id='C', add_APEX=True, add_AACGM=True)

    panel_layouts = [
        [ds['SC_GEO_LAT']],
        [ds['BOUNDARY_FLAG']],
        [ds['SC_GEO_LAT_EB'], ds['SC_GEO_LAT_PB']],
        [ds['SC_QD_LAT_EB'], ds['SC_QD_LAT_PB']],
        [ds['QUALITY_Pa_EB'], ds['QUALITY_Pa_PB']],
        [ds['QUALITY_Sigma_EB'], ds['QUALITY_Sigma_PB']],
    ]

    db.set_layout(panel_layouts=panel_layouts)
    db.draw()
    db.add_title(title='Swarm-{} AOB/FAC Zoom'.format(ds.sat_id), fontsize='medium', append_time=True)
    
    db.save_figure(file_dir=file_dir_figure, file_name='example_AOB_FAC_Swarm-{}_zoom'.format(ds.sat_id), dpi=100, append_time=False)
    db.show()


if __name__ == "__main__":
    test_swarm_AOB_FAC_overview()
    test_swarm_AOB_FAC_zoom()
