import datetime
import matplotlib.pyplot as plt
import numpy as np
import pathlib

import geospacelab.visualization.mpl.dashboards as dashboards

cwd = pathlib.Path(__file__).parent.resolve()
file_dir_figure = cwd / 'figures'
file_dir_figure.mkdir(parents=True, exist_ok=True)

def test_conjunction_with_site():
    """Test Swarm EFI/LP 1B data product
    
    """
    dt_fr = datetime.datetime(2024, 5, 10, 0, )
    dt_to = datetime.datetime(2024, 5, 13, 0, 0)

    site_info = {
        'glat': 69.58, 'glon': 19.23, 'alt': 0. # EISCAT Tromso site
    }
    
    db = dashboards.TSDashboard(
        dt_fr=dt_fr, dt_to=dt_to, figure_config={'figsize': (8, 10)},
        timeline_extra_labels=['GEO_LAT', 'GEO_LON', 'APEX_LAT', 'APEX_LON', 'APEX_MLT',]
        )

    ds_fac = db.dock(
        datasource_contents=['esa_eo', 'swarm', 'l2daily', 'fac_tms'], 
        product_version='latest', # 'latest' (default),
        sat_id='A', add_APEX=False)
    
    conj_list = ds_fac.get_conjunction_with_site(
         glat_site=site_info['glat'], glon_site=site_info['glon'], alt_site=site_info['alt'],
         el_lim=60.,
         print_conj_list=True,
     )
    conj_data = conj_list[1]
    dt_fr_to_show = conj_data['DATETIME_0']
    dt_to_to_show = conj_data['DATETIME_1']
    
    ds_fac.dt_fr = dt_fr_to_show
    ds_fac.dt_to = dt_to_to_show
    ds_fac.time_filter_by_range()
    ds_fac.convert_to_APEX()
    
    ds_efi_lp = db.dock(
        dt_fr=dt_fr_to_show, dt_to=dt_to_to_show,
        datasource_contents=['esa_eo', 'swarm', 'l1b', 'efi_lp'], 
        product_version='latest', # 'latest' (default),
        sat_id='A', add_APEX=True)
    
    ds_mag_lr = db.dock(
        dt_fr=dt_fr_to_show, dt_to=dt_to_to_show,
        datasource_contents=['esa_eo', 'swarm', 'l1b', 'mag_lr'], 
        from_HAPI=True,
        product_version='latest', # 'latest' (default),
        sat_id='A', add_APEX=True)

    
    panel_layouts = [
        [ds_efi_lp['n_e']],
        [ds_efi_lp['n_i']],
        [ds_efi_lp['T_e']],
        [ds_mag_lr['B_res_Model_N'], ds_mag_lr['B_res_Model_E'], ds_mag_lr['B_res_Model_C']],
        [ds_fac['j_r']],
        [ds_fac['j_FA']],
    ]

    db.set_layout(panel_layouts=panel_layouts)
    db.draw(
        dt_fr=dt_fr_to_show, dt_to=dt_to_to_show,
    )
    db.add_title(title='Swarm-{} - EISCAT Tromso conjunction'.format(ds_fac.sat_id), fontsize='medium', append_time=True)

    db.save_figure(file_dir=file_dir_figure, file_name='example_conjuction_with_site_Swarm-{}_Tromso'.format(ds_fac.sat_id), dpi=100, append_time=False)
    db.show()


if __name__ == "__main__":
    test_conjunction_with_site()