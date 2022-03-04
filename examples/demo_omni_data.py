# Licensed under the BSD 3-Clause License
# Copyright (C) 2021 GeospaceLab (geospacelab)
# Author: Lei Cai, Space Physics and Astronomy, University of Oulu

__author__ = "Lei Cai"
__copyright__ = "Copyright 2021, GeospaceLab"
__license__ = "BSD-3-Clause License"
__email__ = "lei.cai@oulu.fi"
__docformat__ = "reStructureText"

import datetime
import geospacelab.express.omni_dashboard as omni

dt_fr = datetime.datetime.strptime('20160321' + '0600', '%Y%m%d%H%M')
dt_to = datetime.datetime.strptime('20160330' + '0600', '%Y%m%d%H%M')

omni_type = 'OMNI2'
omni_res = '1min'
load_mode = 'AUTO'
dashboard = omni.OMNIDashboard(
    dt_fr, dt_to, omni_type=omni_type, omni_res=omni_res, load_mode=load_mode
)

# data can be retrieved in the same way as in Example 1:
dashboard.list_assigned_variables()
B_x_gsm = dashboard.get_variable('B_x_GSM', dataset_index=1)    # Omni dataset index is 1 in the OMNIDashboard. To check other dashboards, use the method "list_datasets()"
print(B_x_gsm)

dashboard.quicklook()

# save figure
dashboard.save_figure()