# Licensed under the BSD 3-Clause License
# Copyright (C) 2021 GeospaceLab (geospacelab)
# Author: Lei Cai, Space Physics and Astronomy, University of Oulu

__author__ = "Lei Cai"
__copyright__ = "Copyright 2021, GeospaceLab"
__license__ = "BSD-3-Clause License"
__email__ = "lei.cai@oulu.fi"
__docformat__ = "reStructureText"


import datetime

import geospacelab.visualization.mpl.dashboards as dashboards
from geospacelab.express.millstonehill_dashboard import MillstoneHillISRDashboard


def example_basic_quicklook():

    # Set the starting and stopping times
    dt_fr = datetime.datetime.strptime('20160314' + '1900', '%Y%m%d%H%M')
    dt_to = datetime.datetime.strptime('20160315' + '0100', '%Y%m%d%H%M')

    # Set the radar parameters
    antenna = 'misa'    # antenna name: 'zenith', or 'misa'
    pulse_code = 'single pulse'     # pulse code: 'single pulse', 'alternating'
    pulse_length = 480              # pulse length: an integer. If not specified,
                                    # the package will search all the pulses in the data file
                                    # and show a message in the console.
    exp_name_pattern = ''  # the key words of the experiment (less than 5 words), the experiment name can be quried using "exp_check".
    exp_check = False               # If True, list all the experiment during the period.
    load_mode = 'AUTO'              # default loading mode.
    dashboard = MillstoneHillISRDashboard(
        dt_fr, dt_to, antenna=antenna, pulse_code=pulse_code, pulse_length=pulse_length,
        exp_name_pattern=exp_name_pattern, exp_check=exp_check,
        load_mode=load_mode
    )
    # Select one or multiple beams shown in the quicklook plot.
    # if commented, all beams will shown following the time sequence.
    dashboard.select_beams(az_el_pairs=[[13.5, 45]])
    # Make quicklook plot
    dashboard.quicklook(depend_MLAT=False)
    # Save the figure
    dashboard.save_figure(file_name="example_mho_basic_quicklook")
    # Display the figure in the screen
    dashboard.show()


def example_combined_datasets():
    # Load multiple datasets and customize the panel layout in the figure

    # Set the starting and stopping times
    dt_fr = datetime.datetime(2016, 3, 14, 18)
    dt_to = datetime.datetime(2016, 3, 15, 2, )

    # Create a dashboard
    dashboard = dashboards.TSDashboard(dt_fr=dt_fr, dt_to=dt_to, figure='new')

    # Dock the first dataset for Millstone Hill ion velocity data.
    ds1 = dashboard.dock(
        datasource_contents=['madrigal', 'isr', 'millstonehill', 'vi'])
    # Dock the second dataset for Millstone Hill gridded data.
    ds2 = dashboard.dock(
        datasource_contents=['madrigal', 'isr', 'millstonehill', 'gridded'])
    # Dock the third dataset for Millstone Hill analyzed data (basic variables).
    ds3 = dashboard.dock(
        datasource_contents=['madrigal', 'isr', 'millstonehill', 'basic'], data_file_type='combined',
        antenna='zenith', pulse_code='single pulse', pulse_length=480)

    # Assign the variables (ion velocities) from the first dataset
    v_i_N = dashboard.assign_variable('v_i_N', dataset=ds1)
    v_i_E = dashboard.assign_variable('v_i_E', dataset=ds1)
    v_i_Z = dashboard.assign_variable('v_i_Z', dataset=ds1)
    E_E = dashboard.assign_variable('E_E', dataset=ds1)
    E_N = dashboard.assign_variable('E_N', dataset=ds1)
    # Set the value limitations for v_i_Z
    v_i_Z.visual.axis[2].lim = [-80, 80]

    # Assign the variables from the third dataset
    n_e = dashboard.assign_variable('n_e', dataset=ds3)

    # Assign the variables from the third dataset
    n_e_grid = dashboard.assign_variable('n_e', dataset=ds2)
    n_e_grid.label = r'$n_e^{gridded}$'

    # Set the panel layout for multiple variables
    layout = [[n_e], [n_e_grid], [v_i_Z], [v_i_E], [v_i_N], [E_E], [E_N]]
    dashboard.set_layout(panel_layouts=layout, hspace=0.1)

    # Generate the figure
    dashboard.draw()
    # Save the figure
    dashboard.save_figure(file_name='example_mho_combined_with_vi', append_time=True)
    # Display the figure
    dashboard.show()
    pass


if __name__ == '__main__':
    example_basic_quicklook()
    # example_combined_datasets()
