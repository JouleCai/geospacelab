import datetime
import matplotlib.pyplot as plt
import numpy as np
import pathlib

import geospacelab.visualization.mpl.dashboards as dashboards

cwd = pathlib.Path(__file__).parent.resolve()
file_dir_figure = cwd / 'figures'
file_dir_figure.mkdir(parents=True, exist_ok=True)

def test_swarm_TEC_TMS_overview():
    """Test Swarm TEC TMS data product
    
    """
    dt_fr = datetime.datetime(2016, 3, 14, 15, 0)
    dt_to = datetime.datetime(2016, 3, 14, 22, 0)

    db = dashboards.TSDashboard(
        dt_fr=dt_fr, dt_to=dt_to, figure_config={'figsize': (8, 12)},
        )

    ds = db.dock(datasource_contents=['esa_eo', 'swarm', 'l2daily', 'tec_tms'], sat_id='A', )

    panel_layouts = [
        [ds['VTEC_ABS']],
        [ds['STEC_ABS']], 
        [ds['STEC_REL']],
        [ds['EL'],],
        [ds['DCB']],
    ]

    db.set_layout(panel_layouts=panel_layouts)
    db.draw()
    db.add_title(title='Swarm-{} TEC TMS Overview'.format(ds.sat_id), fontsize='medium', append_time=True)

    db.save_figure(file_dir=file_dir_figure, file_name='example_TEC_TMS_Swarm-{}_overview'.format(ds.sat_id), dpi=100, append_time=False)
    db.show()

    


if __name__ == "__main__":
    test_swarm_TEC_TMS_overview()
