import datetime
import matplotlib.pyplot as plt

import geospacelab.express.eiscat_dashboard as eiscat


def test_UHF_CP2():
    dt_fr = datetime.datetime.strptime('20161023' + '1600', '%Y%m%d%H%M')
    dt_to = datetime.datetime.strptime('20161023' + '2100', '%Y%m%d%H%M')

    site = 'UHF'
    antenna = 'UHF'
    modulation = '44.4'
    load_mode = 'AUTO'
    data_file_type = 'eiscat-hdf5'

    dashboard = eiscat.EISCATDashboard(dt_fr, dt_to, site=site, antenna=antenna, modulation=modulation,
                                 data_file_type=data_file_type, load_mode=load_mode, status_control=True,
                                 residual_control=True)

    # select beams before assign the variables
    # dashboard.dataset.select_beams(field_aligned=True)
    dashboard.dataset.select_beams(az_el_pairs=[(187.0, 77.7)])
    dashboard.check_beams()

    dashboard.status_mask()

    n_e = dashboard.assign_variable('n_e')
    T_i = dashboard.assign_variable('T_i')
    T_e = dashboard.assign_variable('T_e')
    v_i = dashboard.assign_variable('v_i_los')
    v_i_2 = v_i.clone()
    v_i_2.value = v_i_2.value
    v_i.visual.axis[2].lim=[-100, 100]
    az = dashboard.assign_variable('AZ')
    el = dashboard.assign_variable('EL')
    ptx = dashboard.assign_variable('P_Tx')
    tsys = dashboard.assign_variable('T_SYS_1')
    # T_r = dashboard.dataset.add_variable('T_r', ndim=2)
    # T_r.value = T_e.value / T_i.value
    # T_r.visual.plot_config.style = '2P'
    # T_r.depends = T_i.depends
    # T_r.visual.axis[2].label = 'Te/Ti'
    # T_r.visual.axis[2].unit = ''

    layout = [[n_e], [T_e], [T_i], [v_i], [az, [el], [ptx], [tsys]]]
    dashboard.set_layout(panel_layouts=layout, )
    dashboard.draw()
    dashboard.add_title()
    dashboard.add_panel_labels()
    return dashboard


def test_esr_32m():
    dt_fr = datetime.datetime.strptime('20100616' + '1445', '%Y%m%d%H%M')
    dt_to = datetime.datetime.strptime('20100616' + '1600', '%Y%m%d%H%M')

    site = 'ESR'
    antenna = '42m'
    modulation = ''
    load_mode = 'AUTO'
    data_file_type = 'madrigal-hdf5'

    dashboard = eiscat.EISCATDashboard(dt_fr, dt_to, site=site, antenna=antenna, modulation=modulation,
                                 data_file_type=data_file_type, load_mode=load_mode)

    # select beams before assign the variables
    # dashboard.dataset.select_beams(field_aligned=False)

    n_e = dashboard.assign_variable('n_e')
    T_i = dashboard.assign_variable('T_i')
    T_e = dashboard.assign_variable('T_e')
    v_i = dashboard.assign_variable('v_i_los')
    az = dashboard.assign_variable('AZ')
    el = dashboard.assign_variable('EL')

    n_e.visual.axis[2].lim = [0.5e11, 0.5e12]

    layout = [[n_e], [T_e], [T_i], [v_i], [az, el]]
    dashboard.set_layout(panel_layouts=layout, row_height_scales=[5, 5, 5, 5, 3])
    dashboard.draw()
    dashboard.add_title()
    dashboard.add_panel_labels()
    return dashboard


def test_uhf_cp3():
    dt_fr = datetime.datetime.strptime('20210322' + '2000', '%Y%m%d%H%M')
    dt_to = datetime.datetime.strptime('20210322' + '2300', '%Y%m%d%H%M')

    site = 'UHF'
    antenna = 'UHF'
    modulation = 'ant'
    load_mode = 'AUTO'
    data_file_type = 'madrigal-hdf5'

    dashboard = eiscat.EISCATDashboard(dt_fr, dt_to, site=site, antenna=antenna, modulation=modulation,
                                 data_file_type=data_file_type, load_mode=load_mode)
    dashboard.check_beams()
    # select beams before assign the variables
    # dashboard.dataset.select_beams(field_aligned=False)

    n_e = dashboard.assign_variable('n_e')

    n_e.visual.axis[0].data_res = 180 # time resolution in seconds
    n_e.visual.axis[1].data = '@d.GEO_LAT.value'
    n_e.visual.axis[1].label = 'GLAT'
    n_e.visual.axis[1].unit = 'deg'
    n_e.visual.axis[1].lim = [65, 75]
    T_i = dashboard.assign_variable('T_i')
    T_i.visual.axis[0].data_res = 180
    T_e = dashboard.assign_variable('T_e')
    T_e.visual.axis[0].data_res = 180

    v_i = dashboard.assign_variable('v_i_los')
    v_i.visual.axis[0].data_res = 180
    v_i.visual.axis[1].data = '@d.GEO_LAT.value'
    v_i.visual.axis[1].label = 'GLAT'
    v_i.visual.axis[1].unit = 'deg'
    v_i.visual.axis[1].lim = [65, 75]

    az = dashboard.assign_variable('AZ')
    az.visual.axis[0].data_res = 180
    el = dashboard.assign_variable('EL')
    el.visual.axis[0].data_res = 180

    layout = [[n_e], [T_e], [T_i], [v_i], [az, el]]
    dashboard.set_layout(panel_layouts=layout, row_height_scales=[5, 5, 5, 5, 3])
    dashboard.draw()
    dashboard.add_title()
    dashboard.add_panel_labels()
    return dashboard


def test_vhf_lowel():
    dt_fr = datetime.datetime.strptime('20060903' + '2030', '%Y%m%d%H%M')
    dt_to = datetime.datetime.strptime('20060903' + '2359', '%Y%m%d%H%M')

    site = 'VHF'
    antenna = 'VHF'
    modulation = '60'
    load_mode = 'AUTO'
    data_file_type = 'madrigal-hdf5'

    dashboard = eiscat.EISCATDashboard(dt_fr, dt_to, figure_config={'figsize': (30, 8)}, site=site, antenna=antenna, modulation=modulation,
                                 data_file_type=data_file_type, load_mode=load_mode)
    # select beams before assign the variables
    # dashboard.dataset.select_beams(field_aligned=False)

    n_e = dashboard.assign_variable('n_e')
    n_e_clone = n_e.clone()         # Clone a variable. It has no meaning here, but it may be useful to create panels with the same variable but different visual settings.
    n_e.visual.axis[0].data_res = 180 # time resolution in seconds
    n_e.visual.axis[1].data = '@d.AACGM_LAT.value'
    n_e.visual.axis[1].label = 'MALT'
    n_e.visual.axis[1].unit = 'deg'
    n_e.visual.axis[1].lim = [68, 74.5]
    T_i = dashboard.assign_variable('T_i')
    T_i.visual.axis[0].data_res = 180 # time resolution in seconds
    T_i.visual.axis[1].data = '@d.GEO_LAT.value'
    T_i.visual.axis[1].label = 'GLAT'
    T_i.visual.axis[1].unit = 'deg'
    T_i.visual.axis[1].lim = [71, 75]
    T_e = dashboard.assign_variable('T_e')
    T_e.visual.axis[0].data_res = 180 # time resolution in seconds
    T_e.visual.axis[1].data = '@d.GEO_LAT.value'
    T_e.visual.axis[1].label = 'GLAT'
    T_e.visual.axis[1].unit = 'deg'
    T_e.visual.axis[1].lim = [71, 75]
    v_i = dashboard.assign_variable('v_i_los')
    v_i.visual.axis[0].data_res = 180 # time resolution in seconds
    v_i.visual.axis[1].data = '@d.GEO_LAT.value'
    v_i.visual.axis[1].label = 'GLAT'
    v_i.visual.axis[1].unit = 'deg'
    v_i.visual.axis[1].lim = [None, None]
    az = dashboard.assign_variable('AZ')
    el = dashboard.assign_variable('EL')

    layout = [[n_e], [T_e], [T_i], [v_i], [az, el]]
    dashboard.set_layout(panel_layouts=layout, row_height_scales=[5, 5, 5, 5, 3], right=0.4)
    dashboard.draw()
    dashboard.add_title()
    dashboard.add_panel_labels()

    dt_fr = datetime.datetime.strptime('20001201' + '0400', '%Y%m%d%H%M')
    dt_to = datetime.datetime.strptime('20001201' + '1200', '%Y%m%d%H%M')

    site = 'ESR'
    antenna = '42m'
    modulation = '120'
    load_mode = 'AUTO'
    data_file_type = 'madrigal-hdf5'

    dashboard = dashboard.figure.add_dashboard(dt_fr=dt_fr, dt_to=dt_to, site=site, antenna=antenna, modulation=modulation,
                                       data_file_type=data_file_type, load_mode=load_mode)

    # select beams before assign the variables
    # dashboard.dataset.select_beams(field_aligned=False)

    n_e = dashboard.assign_variable('n_e')
    T_i = dashboard.assign_variable('T_i')
    T_e = dashboard.assign_variable('T_e')
    v_i = dashboard.assign_variable('v_i_los')
    az = dashboard.assign_variable('AZ')
    el = dashboard.assign_variable('EL')

    layout = [[n_e], [T_e], [T_i], [v_i], [az, el]]
    dashboard.set_layout(panel_layouts=layout, row_height_scales=[5, 5, 5, 5, 3], left=0.6)
    dashboard.draw()
    dashboard.add_title()
    dashboard.add_panel_labels()

    return dashboard


if __name__ == "__main__":
    # db = test_vhf_lowel()

    # test_uhf_cp3()
    # test_UHF_CP2()
    test_esr_32m()

    # plt.savefig('eiscat_example.png')
    plt.show()