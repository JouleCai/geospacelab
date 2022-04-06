import datetime

import geospacelab.visualization.mpl.dashboards as dashboards
from geospacelab.express.millstonehill_dashboard import MillstoneHillISRDashboard


def example_basic_quicklook():

    dt_fr = datetime.datetime.strptime('20160315' + '1200', '%Y%m%d%H%M')
    dt_to = datetime.datetime.strptime('20160316' + '1200', '%Y%m%d%H%M')

    antenna = 'misa'
    pulse_code = 'alternating'
    pulse_length = 480
    load_mode = 'AUTO'
    dashboard = MillstoneHillISRDashboard(
        dt_fr, dt_to, antenna=antenna, pulse_code=pulse_code, pulse_length=pulse_length,
    )
    dashboard.select_beams(az_el_pairs=[[-40.5, 45]])
    dashboard.quicklook()
    dashboard.save_figure(file_name="example_basic_quicklook")
    dashboard.show()


def example_combined_datasets():
    dt_fr = datetime.datetime(2016, 3, 14, 18)
    dt_to = datetime.datetime(2016, 3, 15, 2, )
    dashboard = dashboards.TSDashboard(dt_fr=dt_fr, dt_to=dt_to, figure='new')
    ds1 = dashboard.dock(
        datasource_contents=['madrigal', 'isr', 'millstonehill', 'vi'])

    ds2 = dashboard.dock(
        datasource_contents=['madrigal', 'isr', 'millstonehill', 'gridded'])
    ds3 = dashboard.dock(
        datasource_contents=['madrigal', 'isr', 'millstonehill', 'basic'], data_file_type='combined',
        antenna='zenith', pulse_code='single pulse', pulse_length=480)
    v_i_N = dashboard.assign_variable('v_i_N', dataset=ds1)
    v_i_E = dashboard.assign_variable('v_i_E', dataset=ds1)
    v_i_Z = dashboard.assign_variable('v_i_Z', dataset=ds1)
    E_E = dashboard.assign_variable('E_E', dataset=ds1)
    E_N = dashboard.assign_variable('E_N', dataset=ds1)
    v_i_Z.visual.axis[2].lim = [-80, 80]
    n_e = dashboard.assign_variable('n_e', dataset=ds3)
    n_e_grid = dashboard.assign_variable('n_e', dataset=ds2)
    n_e_grid.label = r'$n_e^{gridded}$'

    layout = [[n_e], [n_e_grid], [v_i_Z], [v_i_E], [v_i_N], [E_E], [E_N]]

    dashboard.set_layout(panel_layouts=layout, hspace=0.1)
    dashboard.draw()
    dashboard.save_figure(file_name='example_combined', append_time=True)
    dashboard.show()
    pass


if __name__ == '__main__':
    example_basic_quicklook()
    example_combined_datasets()
