# Licensed under the BSD 3-Clause License
# Copyright (C) 2021 GeospaceLab (geospacelab)
# Author: Lei Cai, Space Physics and Astronomy, University of Oulu

__author__ = "Lei Cai"
__copyright__ = "Copyright 2021, GeospaceLab"
__license__ = "BSD-3-Clause License"
__email__ = "lei.cai@oulu.fi"
__docformat__ = "reStructureText"

import datetime

from geospacelab.express.dmsp_dashboard import DMSPTSDashboard


def show_dmsp():
    sat_id = 'f16'
    dt_c = datetime.datetime(2010, 10, 5, 14, 43)
    diff_minutes = 8
    dt_fr = dt_c - datetime.timedelta(minutes=diff_minutes/2)
    dt_to = dt_c + datetime.timedelta(minutes=diff_minutes/2)

    dashboard = DMSPTSDashboard(
        dt_fr, dt_to, sat_id=sat_id,
    )
    dashboard.quicklook()
    dashboard.save_figure()
    dashboard.show()


if __name__ == '__main__':
    show_dmsp()
