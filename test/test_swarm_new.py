import datetime
import matplotlib.pyplot as plt
import numpy as np

import geospacelab.visualization.mpl.dashboards as dashboards


def test_swarm():
    dt_fr = datetime.datetime(2015, 2, 15, 20, 5)
    dt_to = datetime.datetime(2015, 2, 15, 20, 25)
    time1 = datetime.datetime(2014, 1, 15, 21, 10)
    # specify the file full path

    db = dashboards.TSDashboard(dt_fr=dt_fr, dt_to=dt_to, figure_config={'figsize': (8, 8)})

    db.dock(datasource_contents=['esa_eo', 'swarm', 'advanced', 'efi_lp_hm'], product='LP_HM', sat_id='C', quality_control=False)

    n_e = db.assign_variable('n_e')
    T_e = db.assign_variable('T_e')
    flag = db.assign_variable('QUALITY_FLAG')

    db.set_layout(panel_layouts=[[n_e], [T_e], [flag]])
    db.draw()
    plt.savefig('swarm_example', dpi=300)
    plt.show()


if __name__ == "__main__":
    test_swarm()
