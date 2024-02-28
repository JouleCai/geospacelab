import datetime
import numpy as np
import pathlib

from geospacelab.visualization.mpl.dashboards import TSDashboard


def test_pfisr_fitted():
    dt_fr = datetime.datetime(2022, 2, 1, 18)
    dt_to = datetime.datetime(2022, 2, 1, 23, 59)

    exp_name_pattern = ['plasma', 'line', 'calibration']
    exp_ids = [100278402]
    pulse_code = 'long pulse'
    integration_time = 60.  # in [s]
    beam_id = None
    beam_az = 205.70
    beam_el = 84.5

    db = TSDashboard(dt_fr=dt_fr, dt_to=dt_to)
    ds_pfisr = db.dock(
        datasource_contents=['madrigal', 'isr', 'pfisr', 'fitted'],
        exp_ids=exp_ids,
        exp_name_pattern=exp_name_pattern,
        integration_time=integration_time,
        pulse_code='long pulse',
        beam_id=beam_id,
        beam_az=beam_az,
        beam_el=beam_el
    )

    n_e = ds_pfisr['n_e']
    T_e = ds_pfisr['T_e']
    T_i = ds_pfisr['T_i']
    v_i_los = ds_pfisr['v_i_los']

    panel_layouts = [[n_e], [T_e], [T_i], [v_i_los]]
    db.set_layout(panel_layouts=panel_layouts)

    db.draw()
    db.show()


def test_pfisr_vi():
    
    dt_fr = datetime.datetime(2018, 4, 23, 6)
    dt_to = datetime.datetime(2018, 4, 23, 17, 59)
    
    # exp_name_pattern = ['plasma', 'line', 'calibration']
    exp_ids = [100153185]
    integration_time = 60.  # in [s]
    
    db = TSDashboard(dt_fr=dt_fr, dt_to=dt_to)
    ds_pfisr = db.dock(
        datasource_contents=['madrigal', 'isr', 'pfisr', 'vi'],
        exp_ids = exp_ids,
        # exp_name_pattern=exp_name_pattern,
        # data_file_ext='hdf5',
        integration_time=integration_time,
        )

    v_i_E = ds_pfisr['v_i_E']
    v_i_E_err = ds_pfisr['v_i_E_err']

    v_i_N = ds_pfisr['v_i_N']
    v_i_N_err = ds_pfisr['v_i_N_err']

    v_i_Z = ds_pfisr['v_i_Z']
    v_i_Z_err = ds_pfisr['v_i_Z_err']

    E_E = ds_pfisr['E_E']
    E_E_err = ds_pfisr['E_E_err']

    E_N = ds_pfisr['E_N']
    E_N_err = ds_pfisr['E_N_err']

    E_MAG = ds_pfisr['EF_MAG']
    E_MAG_err = ds_pfisr['EF_MAG_err']

    panel_layouts = [[v_i_E], [v_i_E_err], [v_i_N], [v_i_N_err], [v_i_Z], [v_i_Z_err], [E_E], [E_E_err], [E_N], [E_N_err],]
    panel_layouts = [[E_E], [E_E_err], [E_N], [E_N_err], [E_MAG], [E_MAG_err]]

    db.set_layout(panel_layouts=panel_layouts)

    db.draw()
    db.show()
    
    pass


if __name__ == "__main__":
    test_pfisr_vi()