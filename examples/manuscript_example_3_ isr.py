import datetime
from geospacelab.express.millstonehill_dashboard import MillstoneHillISRDashboard


# Set the starting and stopping times
dt_fr = datetime.datetime.strptime('20150317' + '1200', '%Y%m%d%H%M')
dt_to = datetime.datetime.strptime('20150319' + '1200', '%Y%m%d%H%M')

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
        load_mode=load_mode, figure_config={'figsize': (10, 15)}
    )
# Select one or multiple beams shown in the quicklook plot.
# if commented, all beams will shown following the time sequence.
dashboard.select_beams(az_el_pairs=[[0, 45]])
# Make quicklook plot
dashboard.quicklook(depend_MLAT=True)
# Save the figure
dashboard.save_figure(file_name="manuscript_example_3_mho_basic_quicklook_2")
# Display the figure in the screen
dashboard.show()