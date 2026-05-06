import datetime
import matplotlib.pyplot as plt
import numpy as np
import pathlib

import geospacelab.visualization.mpl.dashboards as dashboards

cwd = pathlib.Path(__file__).parent.resolve()
file_dir_figure = cwd / 'figures'
file_dir_figure.mkdir(parents=True, exist_ok=True)

def test_swarm_WHI_EVT_overview():
    """Test Swarm WHI EVT data product
    
    """
    dt_fr = datetime.datetime(2024, 5, 10, 15, 0)
    dt_to = datetime.datetime(2024, 5, 10, 22, 0)

    db = dashboards.TSDashboard(
        dt_fr=dt_fr, dt_to=dt_to, figure_config={'figsize': (8, 12)},
        )

    ds = db.dock(datasource_contents=['esa_eo', 'swarm', 'l2daily', 'whi_evt'], sat_id='A', )


if __name__ == "__main__":
    test_swarm_WHI_EVT_overview()
