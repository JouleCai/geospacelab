import datetime
import matplotlib.pyplot as plt

import geospacelab.visualization.eiscat_viewer as eiscat


def test_UHF_CP2():
    dt_fr = datetime.datetime.strptime('20210304' + '2300', '%Y%m%d%H%M')
    dt_to = datetime.datetime.strptime('20210305' + '0300', '%Y%m%d%H%M')

    site = 'UHF'
    antenna = 'UHF'
    modulation = 'ant'
    load_mode = 'AUTO'
    data_file_type = 'eiscat-hdf5'

    viewer = eiscat.EISCATViewer(dt_fr, dt_to, site=site, antenna=antenna, modulation=modulation,
                                 data_file_type=data_file_type, load_mode=load_mode)

    # select beams before assign the variables
    # viewer.dataset.select_beams(field_aligned=True)
    viewer.dataset.select_beams(az_el_pairs=[(188.6, 77.7)])

    n_e = viewer.assign_variable('n_e')
    T_i = viewer.assign_variable('T_i')
    T_e = viewer.assign_variable('T_e')
    v_i = viewer.assign_variable('v_i_los')
    az = viewer.assign_variable('az')
    el = viewer.assign_variable('el')

    layout = [[n_e], [T_e], [T_i], [v_i], [az, el]]
    viewer.set_layout(panel_layouts=layout, row_height_scales=[5, 5, 5, 5, 3])
    viewer.draw()
    viewer.add_title()
    viewer.add_panel_labels()
    return viewer


def test_esr_32m():
    dt_fr = datetime.datetime.strptime('20210304' + '2300', '%Y%m%d%H%M')
    dt_to = datetime.datetime.strptime('20210305' + '0300', '%Y%m%d%H%M')

    site = 'ESR'
    antenna = '42m'
    modulation = 'ant'
    load_mode = 'AUTO'
    data_file_type = 'eiscat-hdf5'

    viewer = eiscat.EISCATViewer(dt_fr, dt_to, site=site, antenna=antenna, modulation=modulation,
                                 data_file_type=data_file_type, load_mode=load_mode)

    # select beams before assign the variables
    # viewer.dataset.select_beams(field_aligned=False)

    n_e = viewer.assign_variable('n_e')
    T_i = viewer.assign_variable('T_i')
    T_e = viewer.assign_variable('T_e')
    v_i = viewer.assign_variable('v_i_los')
    az = viewer.assign_variable('az')
    el = viewer.assign_variable('el')

    layout = [[n_e], [T_e], [T_i], [v_i], [az, el]]
    viewer.set_layout(panel_layouts=layout, row_height_scales=[5, 5, 5, 5, 3])
    viewer.draw()
    viewer.add_title()
    viewer.add_panel_labels()
    return viewer


def test_uhf_cp3():
    dt_fr = datetime.datetime.strptime('20210322' + '2000', '%Y%m%d%H%M')
    dt_to = datetime.datetime.strptime('20210322' + '2300', '%Y%m%d%H%M')

    site = 'UHF'
    antenna = 'UHF'
    modulation = 'ant'
    load_mode = 'AUTO'
    data_file_type = 'eiscat-hdf5'

    viewer = eiscat.EISCATViewer(dt_fr, dt_to, site=site, antenna=antenna, modulation=modulation,
                                 data_file_type=data_file_type, load_mode=load_mode)

    # select beams before assign the variables
    # viewer.dataset.select_beams(field_aligned=False)

    n_e = viewer.assign_variable('n_e')
    n_e.visual.axis[0].data_res = 180 # time resolution in seconds
    n_e.visual.axis[1].data = '@d.GEO_LAT.value'
    n_e.visual.axis[1].label = 'GLAT'
    n_e.visual.axis[1].unit = 'deg'
    n_e.visual.axis[1].lim = [65, 75]
    T_i = viewer.assign_variable('T_i')
    T_i.visual.axis[0].data_res = 180
    T_e = viewer.assign_variable('T_e')
    T_e.visual.axis[0].data_res = 180

    v_i = viewer.assign_variable('v_i_los')
    v_i.visual.axis[0].data_res = 180
    v_i.visual.axis[1].data = '@d.GEO_LAT.value'
    v_i.visual.axis[1].label = 'GLAT'
    v_i.visual.axis[1].unit = 'deg'
    v_i.visual.axis[1].lim = [65, 75]

    az = viewer.assign_variable('az')
    az.visual.axis[0].data_res = 180
    el = viewer.assign_variable('el')
    el.visual.axis[0].data_res = 180

    layout = [[n_e], [T_e], [T_i], [v_i], [az, el]]
    viewer.set_layout(panel_layouts=layout, row_height_scales=[5, 5, 5, 5, 3])
    viewer.draw()
    viewer.add_title()
    viewer.add_panel_labels()
    return viewer


def test_vhf_lowel():
    dt_fr = datetime.datetime.strptime('20060903' + '2030', '%Y%m%d%H%M')
    dt_to = datetime.datetime.strptime('20060903' + '2359', '%Y%m%d%H%M')

    site = 'VHF'
    antenna = 'VHF'
    modulation = '60'
    load_mode = 'AUTO'
    data_file_type = 'eiscat-hdf5'

    viewer = eiscat.EISCATViewer(dt_fr, dt_to, site=site, antenna=antenna, modulation=modulation,
                                 data_file_type=data_file_type, load_mode=load_mode)

    # select beams before assign the variables
    # viewer.dataset.select_beams(field_aligned=False)

    n_e = viewer.assign_variable('n_e')
    n_e.visual.axis[0].data_res = 180 # time resolution in seconds
    n_e.visual.axis[1].data = '@d.AACGM_LAT.value'
    n_e.visual.axis[1].label = 'MALT'
    n_e.visual.axis[1].unit = 'deg'
    n_e.visual.axis[1].lim = [68, 74.5]
    T_i = viewer.assign_variable('T_i')
    T_i.visual.axis[0].data_res = 180 # time resolution in seconds
    T_i.visual.axis[1].data = '@d.GEO_LAT.value'
    T_i.visual.axis[1].label = 'GLAT'
    T_i.visual.axis[1].unit = 'deg'
    T_i.visual.axis[1].lim = [71, 75]
    T_e = viewer.assign_variable('T_e')
    T_e.visual.axis[0].data_res = 180 # time resolution in seconds
    T_e.visual.axis[1].data = '@d.GEO_LAT.value'
    T_e.visual.axis[1].label = 'GLAT'
    T_e.visual.axis[1].unit = 'deg'
    T_e.visual.axis[1].lim = [71, 75]
    v_i = viewer.assign_variable('v_i_los')
    v_i.visual.axis[0].data_res = 180 # time resolution in seconds
    v_i.visual.axis[1].data = '@d.GEO_LAT.value'
    v_i.visual.axis[1].label = 'GLAT'
    v_i.visual.axis[1].unit = 'deg'
    v_i.visual.axis[1].lim = [None, None]
    az = viewer.assign_variable('az')
    el = viewer.assign_variable('el')

    layout = [[n_e], [T_e], [T_i], [v_i], [az, el]]
    viewer.set_layout(panel_layouts=layout, row_height_scales=[5, 5, 5, 5, 3])
    viewer.draw()
    viewer.add_title()
    viewer.add_panel_labels()
    return viewer

if __name__ == "__main__":
    test_vhf_lowel()
    test_uhf_cp3()
    test_UHF_CP2()
    test_esr_32m()
    plt.show()