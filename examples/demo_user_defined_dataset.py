# Licensed under the BSD 3-Clause License
# Copyright (C) 2021 GeospaceLab (geospacelab)
# Author: Lei Cai, Space Physics and Astronomy, University of Oulu

__author__ = "Lei Cai"
__copyright__ = "Copyright 2021, GeospaceLab"
__license__ = "BSD-3-Clause License"
__email__ = "lei.cai@oulu.fi"
__docformat__ = "reStructureText"


import numpy as np
import datetime
import geospacelab.datahub as datahub
import geospacelab.visualization.mpl.dashboards as dashboards


def example_add_user_defined_dataset_to_datahub():

    # create a datahub object
    # Optional settings for starting and stopping times.
    dt_fr = datetime.datetime(2015, 9, 1, 0)
    dt_to = datetime.datetime(2015, 9, 1, 12)
    dh = datahub.DataHub(dt_fr=dt_fr, dt_to=dt_to)

    # Add a user-defined dataset object
    ds_user = dh.add_dataset(kind='user-defined')

    # Add several variables that loaded from a user's own script
    # The following case simulate the measurements along a satellite trajectory
    t = np.arange(dt_fr, dt_to, datetime.timedelta(minutes=1)).astype(datetime.datetime)
    t = t.reshape(t.size, 1)
    data_arr_1 = np.random.rand(t.size, 1) * 1e-12      # possible distribution of electron density depending on time

    # add variables to the dataset
    ds_user.add_variable('SC_DATETIME', value=t)
    ds_user.add_variable('n_e', value=data_arr_1)

    print(ds_user['SC_DATETIME'].value)
    print(ds_user['n_e'].value)
    pass


def example_add_user_defined_dataset_to_dashboard():
    # create a time-series dashboard object
    # Optional settings for starting and stopping times.
    dt_fr = datetime.datetime(2015, 9, 1, 0)
    dt_to = datetime.datetime(2015, 9, 1, 12)
    db = dashboards.TSDashboard(dt_fr=dt_fr, dt_to=dt_to)

    # Add a user-defined dataset object
    ds_user = db.add_dataset(kind='user-defined')

    # Add several variables that loaded from a user's own script
    # The following case simulate the measurements along a satellite trajectory
    t = np.arange(dt_fr, dt_to, datetime.timedelta(minutes=1)).astype(datetime.datetime)
    t = t.reshape(t.size, 1)
    data_arr_1 = np.random.rand(t.size, 1) * 1e-12  # possible distribution of electron density depending on time

    # add variables to the dataset
    sc_datetime = ds_user.add_variable('SC_DATETIME', value=t)
    n_e = ds_user.add_variable('n_e', value=data_arr_1)

    # Optional settings for visualization
    n_e.set_depend(0, {'UT': 'SC_DATETIME'})    # link axis-0 to the variable "SC_DATETIME"
    n_e.label = r'$n_e$'        # label to be displayed in the figure
    n_e.unit = 'm-3'
    n_e.unit_label = r'm$^{-3}$'
    n_e.visual.axis[1].lim = [0, 1.2e-12] # set the lim for axis-1 (y-axis)
    n_e.visual.plot_config.style = '1P'  # 1-D plot
    n_e.visual.plot_config.line = {'linestyle': '--'}

    # set the dashboard and panel layouts
    db.set_layout(panel_layouts=[[n_e]])
    db.draw()
    db.show()


if __name__ == "__main__":
    example_add_user_defined_dataset_to_datahub()
    example_add_user_defined_dataset_to_dashboard()
