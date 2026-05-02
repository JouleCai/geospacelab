import datetime
import matplotlib.pyplot as plt
import numpy as np
import pathlib

import geospacelab.visualization.mpl.dashboards as dashboards

cwd = pathlib.Path(__file__).parent.resolve()
file_dir_figure = cwd / 'figures'
file_dir_figure.mkdir(parents=True, exist_ok=True)

def test_swarm_DNS_ACC_overview():
    """Test Swarm DNS/ACC data product
    
    """
    dt_fr = datetime.datetime(2024, 5, 10, 12, 0)
    dt_to = datetime.datetime(2024, 5, 11, 12, 0)

    db = dashboards.TSDashboard(
        dt_fr=dt_fr, dt_to=dt_to, figure_config={'figsize': (8, 8)},
        )

    ds = db.dock(
        datasource_contents=['esa_eo', 'swarm', 'l2daily', 'dns_acc'], 
        sat_id='C',
        product_version='latest',  # Default: "latest", or specify a version like "0201" 
        add_APEX=True, add_AACGM=True)
    
    panel_layouts = [
        [ds['rho_n']],
        [ds['SC_GEO_LAT']],
        [ds['SC_GEO_LON']],
        [ds['SC_GEO_LST']],
    ]

    db.set_layout(panel_layouts=panel_layouts)
    db.draw()
    db.add_title(title='Swarm-{} DNS/ACC Overview'.format(ds.sat_id), fontsize='medium', append_time=True)

    db.save_figure(file_dir=file_dir_figure, file_name='example_DNS_ACC_Swarm-{}_overview'.format(ds.sat_id), dpi=100, append_time=False)
    db.show()
    
def test_swarm_DNS_ACC_zoom():
    """Test Swarm DNS/ACC data product
    
    """
    dt_fr = datetime.datetime(2024, 5, 10, 20, 30)
    dt_to = datetime.datetime(2024, 5, 10, 21, 0)

    db = dashboards.TSDashboard(
        dt_fr=dt_fr, dt_to=dt_to, figure_config={'figsize': (8, 8)},
        timeline_extra_labels=['GEO_LAT', 'GEO_LON', 'AACGM_LAT', 'AACGM_LON', 'AACGM_MLT',] 
        )

    ds = db.dock(datasource_contents=['esa_eo', 'swarm', 'l2daily', 'dns_acc'], sat_id='C', add_APEX=True, add_AACGM=True)

    panel_layouts = [
        [ds['rho_n']],
        [ds['SC_GEO_LAT']],
        [ds['SC_GEO_LON']],
        [ds['SC_GEO_LST']],
    ]

    db.set_layout(panel_layouts=panel_layouts)
    db.draw()
    db.add_title(title='Swarm-{} DNS/ACC Zoom'.format(ds.sat_id), fontsize='medium', append_time=True)
    
    db.save_figure(file_dir=file_dir_figure, file_name='example_DNS_ACC_Swarm-{}_zoom'.format(ds.sat_id), dpi=100, append_time=False)
    db.show()
    

def test_Swarm_DNS_POD_overview():
    """Test Swarm DNS/POD data product
    
    """
    dt_fr = datetime.datetime(2024, 5, 10, 12, 0)
    dt_to = datetime.datetime(2024, 5, 11, 12, 0)

    db = dashboards.TSDashboard(
        dt_fr=dt_fr, dt_to=dt_to, figure_config={'figsize': (8, 8)},
        )

    ds = db.dock(
        datasource_contents=['esa_eo', 'swarm', 'l2daily', 'dns_pod'], 
        sat_id='C',
        product_version='latest', 
        add_APEX=True, add_AACGM=True)
    
    panel_layouts = [
        [ds['rho_n']],
        [ds['rho_n_ORBITMEAN']],
        [ds['SC_GEO_LAT']],
        [ds['SC_GEO_LON']],
        [ds['SC_GEO_LST']],
        [ds['FLAG']],
    ]

    db.set_layout(panel_layouts=panel_layouts)
    db.draw()
    db.add_title(title='Swarm-{} DNS/POD Overview'.format(ds.sat_id), fontsize='medium', append_time=True)

    db.save_figure(file_dir=file_dir_figure, file_name='example_DNS_POD_Swarm-{}_overview'.format(ds.sat_id), dpi=100, append_time=False)
    db.show() 
    

def test_compare_DNS_ACC_POD():
    """Test Swarm DNS/ACC and DNS/POD data products
    
    """
    dt_fr = datetime.datetime(2024, 5, 10, 16, 0)
    dt_to = datetime.datetime(2024, 5, 10, 22, 0)

    db = dashboards.TSDashboard(
        dt_fr=dt_fr, dt_to=dt_to, figure_config={'figsize': (8, 8)},
        # timeline_extra_labels=['GEO_LAT', 'GEO_LON', 'AACGM_LAT', 'AACGM_LON', 'AACGM_MLT',] 
        )

    ds_acc = db.dock(
        datasource_contents=['esa_eo', 'swarm', 'l2daily', 'dns_acc'], 
        product_version='latest',
        sat_id='C', add_APEX=True, add_AACGM=True)
    ds_pod = db.dock(
        datasource_contents=['esa_eo', 'swarm', 'l2daily', 'dns_pod'], 
        product_version='latest',
        sat_id='C', add_APEX=True, add_AACGM=True)

    panel_layouts = [
        [ds_acc['rho_n'], ds_pod['rho_n']],
        [ds_acc['SC_GEO_LAT'],],
        [ds_acc['SC_GEO_LON'],],
        [ds_acc['SC_GEO_LST'],],
    ]
    # Change the default labels to indicate the data source
    ds_acc['rho_n'].visual.axis[2].label = 'DNS/ACC'
    ds_pod['rho_n'].visual.axis[2].label = 'DNS/POD'

    db.set_layout(panel_layouts=panel_layouts)
    db.draw()
    db.add_title(title='Swarm-{} DNS/ACC and DNS/POD Comparison'.format(ds_acc.sat_id), fontsize='medium', append_time=True)
    
    db.save_figure(file_dir=file_dir_figure, file_name='example_DNS_ACC_POD_Swarm-{}_comparison'.format(ds_acc.sat_id), dpi=100, append_time=False)
    db.show()
    
    
def test_compare_POD_versions():
    """Test Swarm DNS/POD data product with different versions
    
    """
    dt_fr = datetime.datetime(2016, 3, 14, 12, 0)
    dt_to = datetime.datetime(2016, 3, 15, 2, 0)

    db = dashboards.TSDashboard(
        dt_fr=dt_fr, dt_to=dt_to, figure_config={'figsize': (8, 8)},
        # timeline_extra_labels=['GEO_LAT', 'GEO_LON', 'AACGM_LAT', 'AACGM_LON', 'AACGM_MLT',] 
        )

    ds_pod_0301 = db.dock(
        datasource_contents=['esa_eo', 'swarm', 'l2daily', 'dns_pod'], 
        product_version='0301',
        sat_id='C', add_APEX=True, add_AACGM=True)
    ds_pod_0201 = db.dock(
        datasource_contents=['esa_eo', 'swarm', 'l2daily', 'dns_pod'], 
        product_version='0201',
        sat_id='C', add_APEX=True, add_AACGM=True)
    ds_pod_0102 = db.dock(
        datasource_contents=['esa_eo', 'swarm', 'l2daily', 'dns_pod'], 
        product_version='0102',
        sat_id='C', add_APEX=True, add_AACGM=True)

    panel_layouts = [
        [ds_pod_0102['rho_n'], ds_pod_0201['rho_n'], ds_pod_0301['rho_n']],
        [ds_pod_0102['rho_n_ORBITMEAN'], ds_pod_0201['rho_n_ORBITMEAN'], ds_pod_0301['rho_n_ORBITMEAN']],
        [ds_pod_0301['SC_GEO_LAT'],],
        [ds_pod_0301['SC_GEO_LON'],],
        [ds_pod_0301['SC_GEO_LST'],],
    ]
    # Change the default labels to indicate the data source
    ds_pod_0301['rho_n'].visual.axis[2].label = 'DNS/POD (0301)'
    ds_pod_0201['rho_n'].visual.axis[2].label = 'DNS/POD (0201)'
    ds_pod_0102['rho_n'].visual.axis[2].label = 'DNS/POD (0102)'

    ds_pod_0301['rho_n_ORBITMEAN'].visual.axis[2].label = 'DNS/POD (0301)'
    ds_pod_0201['rho_n_ORBITMEAN'].visual.axis[2].label = 'DNS/POD (0201)'
    ds_pod_0102['rho_n_ORBITMEAN'].visual.axis[2].label = 'DNS/POD (0102)'

    db.set_layout(panel_layouts=panel_layouts)
    db.draw()
    db.add_title(title='Swarm-{} DNS/POD version Comparison'.format(ds_pod_0301.sat_id), fontsize='medium', append_time=True)
    
    db.save_figure(
        file_dir=file_dir_figure, 
        file_name='example_DNS_POD_Swarm-{}_version_comparison'.format(ds_pod_0301.sat_id), 
        dpi=100, append_time=False)
    db.show()


if __name__ == "__main__":
    # test_swarm_DNS_ACC_overview()
    # test_swarm_DNS_ACC_zoom()
    # test_Swarm_DNS_POD_overview()
    # test_compare_DNS_ACC_POD()
    test_compare_POD_versions()
