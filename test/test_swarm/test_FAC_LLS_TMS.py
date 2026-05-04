import datetime
import matplotlib.pyplot as plt
import numpy as np
import pathlib

import geospacelab.visualization.mpl.dashboards as dashboards

cwd = pathlib.Path(__file__).parent.resolve()
file_dir_figure = cwd / 'figures'
file_dir_figure.mkdir(parents=True, exist_ok=True)

def test_swarm_FAC_LLS_overview():
    """Test Swarm FAC LLS data product
    
    """
    dt_fr = datetime.datetime(2016, 3, 14, 15, 0)
    dt_to = datetime.datetime(2016, 3, 14, 22, 0)
    
    db = dashboards.TSDashboard(
        dt_fr=dt_fr, dt_to=dt_to, figure_config={'figsize': (8, 8)},
        )

    ds_lls = db.dock(datasource_contents=['esa_eo', 'swarm', 'l2daily', 'fac_lls_dual'], add_APEX=True,) # sat_id is fixed to 'AC' for LLS Dual product

    panel_layouts = [
        [ds_lls['j_r']], 
        [ds_lls['j_FA']],
        [ds_lls['FLAG_BIN_AUX'],],
    ]

    db.set_layout(panel_layouts=panel_layouts)
    db.draw()
    db.add_title(title='Swarm-{} FAC Overview'.format(ds_lls.sat_id), fontsize='medium', append_time=True)

    db.save_figure(file_dir=file_dir_figure, file_name='example_FAC_LLS_Swarm-{}_overview'.format(ds_lls.sat_id), dpi=100, append_time=False)
    db.show()
    
def test_swarm_FAC_LLS_zoom():
    """Test Swarm FAC LLS data products
    
    """
    dt_fr = datetime.datetime(2016, 3, 14, 20, 15)
    dt_to = datetime.datetime(2016, 3, 14, 20, 40)

    db = dashboards.TSDashboard(
        dt_fr=dt_fr, dt_to=dt_to, figure_config={'figsize': (8, 8)},
        timeline_extra_labels=['GEO_LAT', 'GEO_LON', 'APEX_LAT', 'APEX_LON', 'APEX_MLT',]  # Not applicable for very scattered data points like AEJ_PBS peaks
        )

    ds_lls = db.dock(datasource_contents=['esa_eo', 'swarm', 'l2daily', 'fac_lls_dual'], add_APEX=True,)

    panel_layouts = [
        [ds_lls['j_r']], 
        [ds_lls['j_FA']],
        [ds_lls['FLAG_BIN_AUX'],],
    ]

    db.set_layout(panel_layouts=panel_layouts)
    db.draw()
    db.add_title(title='Swarm-{} FAC LLS Zoom'.format(ds_lls.sat_id), fontsize='medium', append_time=True)
    
    db.save_figure(file_dir=file_dir_figure, file_name='example_FAC_LLS_Swarm-{}_zoom'.format(ds_lls.sat_id), dpi=100, append_time=False)
    db.show()
    

def test_swarm_FAC_TMS_zoom():
    """Test Swarm FAC TMS data product
    
    """
    dt_fr = datetime.datetime(2016, 3, 14, 20, 21)
    dt_to = datetime.datetime(2016, 3, 14, 20, 22)

    db = dashboards.TSDashboard(
        dt_fr=dt_fr, dt_to=dt_to, figure_config={'figsize': (8, 12)},
        timeline_extra_labels=['GEO_LAT', 'GEO_LON', 'APEX_LAT', 'APEX_LON', 'APEX_MLT',]  # Not applicable for very scattered data points like AEJ_PBS peaks
        )

    ds_tms_dual = db.dock(datasource_contents=['esa_eo', 'swarm', 'l2daily', 'fac_tms_dual'], add_APEX=True,)
    
    ds_tms_A = db.dock(datasource_contents=['esa_eo', 'swarm', 'l2daily', 'fac_tms'], sat_id='A', add_APEX=True,)
    ds_tms_C = db.dock(datasource_contents=['esa_eo', 'swarm', 'l2daily', 'fac_tms'], sat_id='C', add_APEX=True,)

    panel_layouts = [
        [ds_tms_dual['j_r'], ds_tms_A['j_r'], ds_tms_C['j_r'],],
        [ds_tms_dual['j_FA'], ds_tms_A['j_FA'], ds_tms_C['j_FA'],],
        [ds_tms_dual['FLAG_DIGIT_AUX']], 
        [ds_tms_A['FLAG_DIGIT_AUX']], 
        [ds_tms_C['FLAG_DIGIT_AUX'],],
        [ds_tms_dual['FLAG_F_DIGIT_AUX'],],
        [ds_tms_dual['FLAG_B_DIGIT_AUX'],],
        [ds_tms_dual['FLAG_q_DIGIT_AUX'],],
    ]
    
    ds_tms_dual['j_r'].visual.axis[1].label = 'j_r'
    ds_tms_dual['j_FA'].visual.axis[1].label = 'j_FA'
    ds_tms_dual['j_r'].visual.axis[2].label = 'Sw-AC'
    ds_tms_A['j_r'].visual.axis[2].label = 'Sw-A'
    ds_tms_C['j_r'].visual.axis[2].label = 'Sw-C'
    
    ds_tms_dual['j_FA'].visual.axis[2].label = 'Sw-AC'
    ds_tms_A['j_FA'].visual.axis[2].label = 'Sw-A'
    ds_tms_C['j_FA'].visual.axis[2].label = 'Sw-C'
    
    ds_tms_dual['FLAG_DIGIT_AUX'].visual.axis[2].label = 'FLAG Sw-AC'
    ds_tms_A['FLAG_DIGIT_AUX'].visual.axis[2].label = 'FLAG Sw-A'
    ds_tms_C['FLAG_DIGIT_AUX'].visual.axis[2].label = 'FLAG Sw-C'
    
    ds_tms_dual['FLAG_F_DIGIT_AUX'].visual.axis[2].label = 'FLAG F Sw-AC'
    ds_tms_dual['FLAG_B_DIGIT_AUX'].visual.axis[2].label = 'FLAG B Sw-AC'
    ds_tms_dual['FLAG_q_DIGIT_AUX'].visual.axis[2].label = 'FLAG q Sw-AC'

    db.set_layout(panel_layouts=panel_layouts)
    db.draw()
    db.add_title(title='Swarm-{} FAC TMS Zoom'.format(ds_tms_dual.sat_id), fontsize='medium', append_time=True)
    
    db.save_figure(file_dir=file_dir_figure, file_name='example_FAC_TMS_Swarm-{}_zoom'.format(ds_tms_dual.sat_id), dpi=100, append_time=False)
    db.show()


if __name__ == "__main__":
    # test_swarm_FAC_LLS_overview()
    # test_swarm_FAC_LLS_zoom()
    test_swarm_FAC_TMS_zoom()