import datetime
import matplotlib.pyplot as plt
import numpy as np

import geospacelab.visualization.mpl.ts_viewer as ts_viewer


def test_swarm():
    dt_fr = datetime.datetime(2016, 3, 14, 8)
    dt_to = datetime.datetime(2016, 3, 14, 23, 59)
    time1 = datetime.datetime(2016, 3, 14, 21, 10)
    # specify the file full path

    viewer = ts_viewer.TimeSeriesViewer(dt_fr=dt_fr, dt_to=dt_to, figure_config={'figsize': (8, 8)})

    viewer.dock(datasource_contents=['esa_eo', 'swarm', 'advanced', 'efi_tct'], product='TCT02')

    plt.savefig('swarm_example', dpi=300)
    plt.show()


if __name__ == "__main__":
    test_swarm()
