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
import geospacelab.toolbox.utilities.pylogging as mylog


class DMSPTSDashboard(TSDashboard):
    def __init__(
            self,
            dt_fr: datetime.datetime,
            dt_to: datetime.datetime,
            sat_id: str = None,
            load_mode: str = 'AUTO',
            figure: str = 'new',
            figure_config: dict = {'figsize': (10, 10)},
            timeline_extra_labels: list = None,
            **kwargs
    ):
        if timeline_extra_labels is None:
            timeline_extra_labels = ['GEO_LAT', 'GEO_LON', 'AACGM_LAT', 'AACGM_MLT']
        super().__init__(dt_fr=dt_fr, dt_to=dt_to,
                         figure=figure, figure_config=figure_config,
                         timeline_extra_labels=timeline_extra_labels)
        self.dataset_s1 = self.dock(datasource_contents=['madrigal', 'satellites', 'dmsp', 's1'], sat_id=sat_id, load_mode=load_mode, **kwargs)
        self.dataset_e = self.dock(datasource_contents=['madrigal', 'satellites', 'dmsp', 'e'], sat_id=sat_id, load_mode=load_mode, **kwargs)
        self.dataset_s4 = self.dock(datasource_contents=['madrigal', 'satellites', 'dmsp', 's4'], sat_id=sat_id, load_mode=load_mode, **kwargs)
        # ds_1.list_all_variables()
        self.title = kwargs.pop('title', ', '.join([self.dataset_s1.database, self.dataset_s1.facility, self.dataset_s1.sat_id]))

    def save_figure(self, file_name=None, file_dir=None, append_time=True, **kwargs):
        if file_name is None:
            file_name = kwargs.pop('file_name', self.title.replace(', ', '_'))
        super().save_figure(file_name=file_name, file_dir=file_dir, append_time=append_time, **kwargs)

    def add_title(self, x=0.5, y=1.06, title=None, append_time=True, **kwargs):
        if title is None:
            title = self.title
        super().add_title(x=x, y=y, title=title, append_time=append_time, **kwargs)

    def quicklook(self):

        n_e = self.assign_variable('n_e', dataset=self.dataset_s1)
        v_i_H = self.assign_variable('v_i_H', dataset=self.dataset_s1)
        v_i_V = self.assign_variable('v_i_V', dataset=self.dataset_s1)
        d_B_D = self.assign_variable('d_B_D', dataset=self.dataset_s1)
        d_B_P = self.assign_variable('d_B_P', dataset=self.dataset_s1)
        d_B_F = self.assign_variable('d_B_F', dataset=self.dataset_s1)

        JE_e = self.assign_variable('JE_e', dataset=self.dataset_e)
        JE_i = self.assign_variable('JE_i', dataset=self.dataset_e)
        jE_e = self.assign_variable('jE_e', dataset=self.dataset_e)
        jE_i = self.assign_variable('jE_i', dataset=self.dataset_e)
        E_e_MEAN = self.assign_variable('E_e_MEAN', dataset=self.dataset_e)
        E_i_MEAN = self.assign_variable('E_i_MEAN', dataset=self.dataset_e)

        T_i = self.assign_variable('T_i', dataset=self.dataset_s4)
        T_e = self.assign_variable('T_e', dataset=self.dataset_s4)
        c_O_p = self.assign_variable('COMP_O_p', dataset=self.dataset_s4)

        self.list_assigned_variables()
        self.list_datasets()

        layout = [
            [v_i_H, v_i_V],
            [d_B_P, d_B_D, d_B_F],
            [E_e_MEAN, E_i_MEAN],
            [JE_e, JE_i],
            [jE_e],
            [jE_i],
            [n_e, [c_O_p]],
            [T_e, T_i],
        ]
        self.set_layout(panel_layouts=layout, hspace=0.1)

        self.draw()
        self.add_title()
        self.add_panel_labels()


def example():

    dt_fr = datetime.datetime.strptime('20100908' + '2034', '%Y%m%d%H%M')
    dt_to = datetime.datetime.strptime('20100908' + '2036', '%Y%m%d%H%M')

    dashboard = DMSPTSDashboard(
        dt_fr, dt_to, sat_id='F18',
    )
    dashboard.quicklook()
    dashboard.show()


if __name__ == '__main__':
    example()

