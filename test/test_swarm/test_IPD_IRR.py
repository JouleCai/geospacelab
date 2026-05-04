import datetime
import matplotlib.pyplot as plt
import numpy as np
import pathlib

import geospacelab.visualization.mpl.dashboards as dashboards

cwd = pathlib.Path(__file__).parent.resolve()
file_dir_figure = cwd / 'figures'
file_dir_figure.mkdir(parents=True, exist_ok=True)

def test_swarm_IPD_IRR_overview():
    """Test Swarm IPD IRR data product
    
    """
    dt_fr = datetime.datetime(2016, 3, 14, 12, 0)
    dt_to = datetime.datetime(2016, 3, 15, 12, 59)

    db = dashboards.TSDashboard(
        dt_fr=dt_fr, dt_to=dt_to, figure_config={'figsize': (8, 12)},
        )

    ds = db.dock(datasource_contents=['esa_eo', 'swarm', 'l2daily', 'ipd_irr'], sat_id='A', add_APEX=True,)

    panel_layouts = [
        [ds['n_e_BKG'], ds['n_e_FRG'], ds['n_e'], ],
        [ds['T_e']],
        [ds['GRAD_n_e_PCP_EDGE'], ds['GRAD_n_e_20km'], ds['GRAD_n_e_50km'], ds['GRAD_n_e_100km'], ],
        [ds['ROD'], ds['RODI_20s'], ds['RODI_10s']],
        [ds['d_n_e_40s'], ds['d_n_e_20s'], ds['d_n_e_10s']],
        [ds['VTEC_MEDIAN'], [ds['VTEC_STD']]],
        [ds['ROT_MEDIAN'], ds['ROTI_10s_MEDIAN'], ds['ROTI_20s_MEDIAN']],
        [ds['Ionosphere_Region'], ds['FLAG_PCP']],
        [ds['IPIR_INDEX'], ],
    ]

    db.set_layout(panel_layouts=panel_layouts)
    db.draw()
    db.add_title(title='Swarm-{} IPD IRR Overview'.format(ds.sat_id), fontsize='medium', append_time=True)

    db.save_figure(file_dir=file_dir_figure, file_name='example_IPD_IRR_Swarm-{}_overview'.format(ds.sat_id), dpi=100, append_time=False)
    db.show()
    
def test_swarm_IPD_IRR_zoom():
    """Test Swarm IPD IRR data product
    
    """
    dt_fr = datetime.datetime(2016, 3, 15, 18, 16)
    dt_to = datetime.datetime(2016, 3, 15, 18, 23)

    db = dashboards.TSDashboard(
        dt_fr=dt_fr, dt_to=dt_to, figure_config={'figsize': (8, 12)},
        timeline_extra_labels=['GEO_LAT', 'GEO_LON', 'APEX_LAT', 'APEX_LON', 'APEX_MLT',]  # Not applicable for very scattered data points like AEJ_PBS peaks
        )

    ds = db.dock(datasource_contents=['esa_eo', 'swarm', 'l2daily', 'ipd_irr'], sat_id='A', add_APEX=True,)

    panel_layouts = [
        [ds['n_e_BKG'], ds['n_e_FRG'], ds['n_e'],],
        [ds['T_e']],
        [ds['GRAD_n_e_PCP_EDGE'], ds['GRAD_n_e_20km'], ds['GRAD_n_e_50km'], ds['GRAD_n_e_100km'], ],
        [ds['ROD'], ds['RODI_20s'], ds['RODI_10s']],
        [ds['d_n_e_40s'], ds['d_n_e_20s'], ds['d_n_e_10s']],
        [ds['VTEC_MEDIAN'], [ds['VTEC_STD']]],
        [ds['ROT_MEDIAN'], ds['ROTI_10s_MEDIAN'], ds['ROTI_20s_MEDIAN']],
        [ds['Ionosphere_Region'], ds['FLAG_PCP']],
        [ds['IPIR_INDEX'], ],
    ]

    db.set_layout(panel_layouts=panel_layouts)
    db.draw()
    db.add_title(title='Swarm-{} IPD IRR Zoom'.format(ds.sat_id), fontsize='medium', append_time=True)
    
    db.save_figure(file_dir=file_dir_figure, file_name='example_IPD_IRR_Swarm-{}_zoom'.format(ds.sat_id), dpi=100, append_time=False)
    db.show()
    


if __name__ == "__main__":
    test_swarm_IPD_IRR_overview()
    test_swarm_IPD_IRR_zoom()
