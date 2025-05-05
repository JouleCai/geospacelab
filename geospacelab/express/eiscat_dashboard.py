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


class EISCATDashboard(TSDashboard):
    def __init__(self, dt_fr, dt_to, **kwargs):
        kwargs.setdefault('load_mode', 'AUTO')

        figure = kwargs.pop('figure', 'new')
        figure_config = kwargs.pop('figure_config', {'figsize': (10, 10)})
        timeline_extra_labels = kwargs.pop('timeline_extra_labels', None)
        super().__init__(dt_fr=dt_fr, dt_to=dt_to, figure=figure, figure_config=figure_config, timeline_extra_labels=timeline_extra_labels)
        self.dock(datasource_contents=['madrigal', 'isr', 'eiscat'], **kwargs)
        self.host_dataset.load_data(load_mode=kwargs['load_mode'])
        # ds_1.list_all_variables()
        self.title = kwargs.pop('title', ', '.join([self.host_dataset.facility,
                                                    self.host_dataset.site, self.host_dataset.pulse_code,
                                                    self.host_dataset.scan_mode, self.host_dataset.modulation]))

    def status_mask(self, bad_status=None):
        self.host_dataset.status_mask(bad_status=bad_status)

    def residual_mask(self, residual_lim=None):
        self.host_dataset.residual_mask(residual_lim=residual_lim)

    def outlier_mask(self, condition, fill_value=None):
        self.host_dataset.outlier_mask(condition, fill_value=fill_value)

    def select_beams(self, field_aligned=False, az_el_pairs=None, error_az=2., error_el=2.):
        self.host_dataset.select_beams(field_aligned=field_aligned, az_el_pairs=az_el_pairs,
                                       error_az=error_az, error_el=error_el)

    def list_eiscat_variables(self):
        self.host_dataset.list_all_variables()

    def check_beams(self, error=0.5, logging=True, full_sequence=False):
        import numpy as np
        azV = self.host_dataset['AZ']
        elV = self.host_dataset['EL']
        az_arr = np.round(azV.value, decimals=1)
        el_arr = np.round(elV.value, decimals=1)
        beams = np.array([[az_arr[0, 0], el_arr[0, 0]]])
        beams_counts = [1]
        beams_sequence_inds = [[0]]
        beam_array = np.hstack((az_arr[1:, :], el_arr[1:, :]))
        for ind in range(beam_array.shape[0]):
            tile_beam = np.tile(beam_array[ind], (beams.shape[0], 1))
            diff = np.abs(beams - tile_beam)
            if abs(beam_array[ind, 1] - 90) < error:
                el_is_90 = np.where(np.abs(beams[:, 1] - 90) < error)[0]
                if list(el_is_90):
                    diff[el_is_90, 0] = 0
            ind_beam = np.where(np.all(diff < error, axis=1))
            if not list(ind_beam[0]):
                beams = np.vstack((beams, beam_array[ind, :]))
                beams_counts.append(1)
                beams_sequence_inds.append([ind+1])
            else:
                beams_counts[ind_beam[0][0]] = beams_counts[ind_beam[0][0]] + 1
                beams_sequence_inds[ind_beam[0][0]].append(ind+1)
            # elif len(ind_beam[0]) == 1:
            #     beams_counts[ind_beam[0][0]] = beams_counts[ind_beam[0][0]] + 1
            #     beams_sequence_inds[ind_beam[0][0]].append(ind+1)
            # else:
            #     print(beams)
            #     print(beam_array[ind])
            #     raise ValueError(f"Several beams have the similar az and el angles, which cannot be identified." +
            #                      " Try to set the error of angles with a larger value! Currently error={error}")
        beams_counts = np.array(beams_counts)
        beams_sequence_inds = np.array(beams_sequence_inds, dtype=object)
        count_ind = np.argsort(-beams_counts)
        beams = beams[count_ind, :]
        beams_counts = beams_counts[count_ind]
        beams_sequence_inds = beams_sequence_inds[count_ind]
        if logging:
            label = self.host_dataset.label()
            mylog.simpleinfo.info("Dataset: {}".format(label))
            mylog.simpleinfo.info("Listing all the beams ...")
            mylog.simpleinfo.info('{:^20s}{:^20s}{:^20s}{:80s}'.format('No.', '(az, el)', 'Counts', 'Sequence indices'))
            for ind in range(beams.shape[0]):
                if full_sequence:
                    sequence_str = repr(beams_sequence_inds[ind])
                elif len(beams_sequence_inds[ind]) < 10:
                    sequence_str = repr(beams_sequence_inds[ind])
                else:
                    sequence_str = repr(beams_sequence_inds[ind][:10]).replace(']', ', ...]')
                mylog.simpleinfo.info(
                    '{:^20d}{:^20s}{:^20d}{:80s}'.format(
                        ind+1, f"({'{:.1f}'.format(beams[ind, 0])}, {'{:.1f}'.format(beams[ind, 1])})",
                        beams_counts[ind], sequence_str
                    )
                )

        return beams, beams_counts, beams_sequence_inds

    def save_figure(self, file_name=None, file_dir=None, append_time=True, **kwargs):
        if file_name is None:
            file_name = '_'.join([s.strip() for s in self.title.split(',')])
        super().save_figure(file_name=file_name, file_dir=file_dir, append_time=append_time, **kwargs)

    def add_title(self, x=0.5, y=1.06, title=None, append_time=True, **kwargs):
        if title is None:
            title = self.title
        super().add_title(x=x, y=y, title=title, append_time=append_time, **kwargs)

    def quicklook(self, time_res=None):
        n_e = self.assign_variable('n_e')
        T_i = self.assign_variable('T_i')
        T_e = self.assign_variable('T_e')
        v_i = self.assign_variable('v_i_los')
        az = self.assign_variable('AZ')
        az.visual.axis[1].label = '@v.label'
        az.visual.axis[1].unit = '@v.unit_label'
        el = self.assign_variable('EL')
        el.visual.axis[1].label = '@v.label'
        el.visual.axis[1].unit = '@v.unit_label'
        ptx = self.assign_variable('P_Tx')
        ptx.visual.axis[1].label = '@v.label'
        ptx.visual.axis[1].unit = '@v.unit_label'
        tsys = self.assign_variable('T_SYS_1')
        tsys.visual.axis[1].label = '@v.label'
        tsys.visual.axis[1].unit = '@v.unit_label'
        self.list_assigned_variables()
        self.list_datasets()
        self.check_beams()

        layout = [[n_e], [T_e], [T_i], [v_i], [az, [el], [ptx], [tsys]]]
        self.set_layout(panel_layouts=layout, row_height_scales=[5, 5, 5, 5, 3], hspace=0.1)
        # plt.style.use('dark_background')
        # dt_fr_1 = datetime.datetime.strptime('20201209' + '1300', '%Y%m%d%H%M')
        # dt_to_1 = datetime.datetime.strptime('20201210' + '1200', '%Y%m%d%H%M')

        self.draw(time_res=time_res)
        self.add_title()
        self.add_panel_labels()


def example():

    dt_fr = datetime.datetime.strptime('20201209' + '1800', '%Y%m%d%H%M')
    dt_to = datetime.datetime.strptime('20201210' + '0600', '%Y%m%d%H%M')

    site = 'UHF'
    antenna = 'UHF'
    modulation = '60'
    load_mode = 'AUTO'
    dashboard = EISCATDashboard(
        dt_fr, dt_to, site=site, antenna=antenna, modulation=modulation, load_mode='AUTO',
        data_file_type="madrigal-hdf5"
    )
    dashboard.quicklook()

    # viewer.save_figure() # comment this if you need to run the following codes
    # viewer.show()   # comment this if you need to run the following codes.

    """
    As the viewer is an instance of the class EISCATViewer, which is a heritage of the class Datahub.
    The variables can be retrieved in the same ways as shown in Example 1. 
    """
    n_e = dashboard.assign_variable('n_e')
    print(n_e.value)

    """
    Several marking tools (vertical lines, shadings, and top bars) can be added as the overlays 
    on the top of the quicklook plot.
    """
    # add vertical line
    dt_fr_2 = datetime.datetime.strptime('20201209' + '2030', "%Y%m%d%H%M")
    dt_to_2 = datetime.datetime.strptime('20201210' + '0130', "%Y%m%d%H%M")
    dashboard.add_vertical_line(dt_fr_2, bottom_extend=0, top_extend=0.02, label='Line 1', label_position='top')
    # add shading
    dashboard.add_shading(dt_fr_2, dt_to_2, bottom_extend=0, top_extend=0.02, label='Shading 1', label_position='top')
    # add top bar
    dt_fr_3 = datetime.datetime.strptime('20201210' + '0130', "%Y%m%d%H%M")
    dt_to_3 = datetime.datetime.strptime('20201210' + '0430', "%Y%m%d%H%M")
    dashboard.add_top_bar(dt_fr_3, dt_to_3, bottom=0., top=0.02, label='Top bar 1')

    # save figure
    dashboard.save_figure()
    # show on screen
    dashboard.show()


if __name__ == '__main__':
    example()

