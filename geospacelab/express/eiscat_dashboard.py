# Licensed under the BSD 3-Clause License
# Copyright (C) 2021 GeospaceLab (geospacelab)
# Author: Lei Cai, Space Physics and Astronomy, University of Oulu

__author__ = "Lei Cai"
__copyright__ = "Copyright 2021, GeospaceLab"
__license__ = "BSD-3-Clause License"
__email__ = "lei.cai@oulu.fi"
__docformat__ = "reStructureText"


import datetime

from geospacelab.visualization.mpl.dashboards import TSDashboard


class EISCATDashboard(TSDashboard):
    def __init__(self, dt_fr, dt_to, **kwargs):
        figure = kwargs.pop('figure', 'new')
        super().__init__(dt_fr=dt_fr, dt_to=dt_to, figure=figure)
        ds_1 = self.dock(datasource_contents=['madrigal', 'eiscat'], **kwargs)
        ds_1.load_data(load_mode=kwargs['load_mode'])
        # ds_1.list_all_variables()
        self.title = kwargs.pop('title', ', '.join([ds_1.facility, ds_1.site, ds_1.pulse_code, ds_1.scan_mode, ds_1.modulation]))

    @property
    def dataset(self):
        return self.datasets[1]

    def status_mask(self, bad_status=None):
        self.dataset.status_mask(bad_status=bad_status)

    def residual_mask(self, residual_lim=None):
        self.dataset.residual_mask(residual_lim=residual_lim)

    def outlier_mask(self, condition, fill_value=None):
        self.dataset.outlier_mask(condition, fill_value=fill_value)

    def select_beams(self, field_aligned=False, az_el_pairs=None):
        self.dataset.select_beams(field_aligned=field_aligned, az_el_pairs=az_el_pairs)

    def list_all_variables(self):
        self.datasets[1].list_all_variables()

    def save_figure(self, **kwargs):
        file_name = kwargs.pop('file_name', self.title.replace(', ', '_'))
        super().save_figure(file_name=file_name, **kwargs)

    def add_title(self, **kwargs):
        title = kwargs.pop('title', self.title)
        super().add_title(x=0.5, y=1.06, title=title)

    def quicklook(self):
        n_e = self.assign_variable('n_e')
        T_i = self.assign_variable('T_i')
        T_e = self.assign_variable('T_e')
        v_i = self.assign_variable('v_i_los')
        az = self.assign_variable('AZ')
        el = self.assign_variable('EL')
        self.list_assigned_variables()
        self.list_datasets()

        layout = [[n_e], [T_e], [T_i], [v_i], [az, [el]]]
        self.set_layout(panel_layouts=layout, row_height_scales=[5, 5, 5, 5, 3], hspace=0.1)
        # plt.style.use('dark_background')
        # dt_fr_1 = datetime.datetime.strptime('20201209' + '1300', '%Y%m%d%H%M')
        # dt_to_1 = datetime.datetime.strptime('20201210' + '1200', '%Y%m%d%H%M')

        self.draw()
        self.add_title()
        self.add_panel_labels()


def example():

    dt_fr = datetime.datetime.strptime('20201209' + '1800', '%Y%m%d%H%M')
    dt_to = datetime.datetime.strptime('20201210' + '0600', '%Y%m%d%H%M')

    site = 'UHF'
    antenna = 'UHF'
    modulation = '60'
    load_mode = 'AUTO'
    viewer = EISCATViewer(
        dt_fr, dt_to, site=site, antenna=antenna, modulation=modulation, load_mode='AUTO'
    )
    viewer.quicklook()

    # viewer.save_figure() # comment this if you need to run the following codes
    # viewer.show()   # comment this if you need to run the following codes.

    """
    As the viewer is an instance of the class EISCATViewer, which is a heritage of the class Datahub.
    The variables can be retrieved in the same ways as shown in Example 1. 
    """
    n_e = viewer.assign_variable('n_e')
    print(n_e.value)

    """
    Several marking tools (vertical lines, shadings, and top bars) can be added as the overlays 
    on the top of the quicklook plot.
    """
    # add vertical line
    dt_fr_2 = datetime.datetime.strptime('20201209' + '2030', "%Y%m%d%H%M")
    dt_to_2 = datetime.datetime.strptime('20201210' + '0130', "%Y%m%d%H%M")
    viewer.add_vertical_line(dt_fr_2, bottom_extend=0, top_extend=0.02, label='Line 1', label_position='top')
    # add shading
    viewer.add_shading(dt_fr_2, dt_to_2, bottom_extend=0, top_extend=0.02, label='Shading 1', label_position='top')
    # add top bar
    dt_fr_3 = datetime.datetime.strptime('20201210' + '0130', "%Y%m%d%H%M")
    dt_to_3 = datetime.datetime.strptime('20201210' + '0430', "%Y%m%d%H%M")
    viewer.add_top_bar(dt_fr_3, dt_to_3, bottom=0., top=0.02, label='Top bar 1')

    # save figure
    # viewer.save_figure()
    # show on screen
    viewer.show()


if __name__ == '__main__':
    example()
