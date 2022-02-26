import datetime
import matplotlib.pyplot as plt
import numpy as np

import geospacelab.visualization.mpl.dashboards as dashboards


def test_swarm():
    dt_fr = datetime.datetime(2015, 9, 8, 18)
    dt_to = datetime.datetime(2015, 9, 8, 18, 5)
    time1 = datetime.datetime(2022, 1, 15, 21, 10)
    # specify the file full path

    db = dashboards.TSDashboard(dt_fr=dt_fr, dt_to=dt_to, figure='off')

    # db.dock(datasource_contents=['esa_eo', 'swarm', 'advanced', 'efi_lp_hm'], product='LP_HM', sat_id='C', quality_control=False)

    ds1 = db.dock(datasource_contents=['madrigal', 'dmsp', 's1'], sat_id='F18')

    ds = db.dock(datasource_contents=['madrigal', 'dmsp', 's4'], sat_id='F18')

    ds = db.dock(datasource_contents=['madrigal', 'dmsp', 'e'], sat_id='F18')
    # b_1 = db.assign_variable('')


if __name__ == "__main__":
    test_swarm()
