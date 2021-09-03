import datetime
import geospacelab.express.eiscat_viewer as eiscat

dt_fr = datetime.datetime.strptime('20201209' + '1800', '%Y%m%d%H%M')
dt_to = datetime.datetime.strptime('20201210' + '0600', '%Y%m%d%H%M')

# check the eiscat-hdf5 filename from the EISCAT schedule page, e.g., "EISCAT_2020-12-10_beata_60@uhfa.hdf5"
site = 'UHF'
antenna = 'UHF'
modulation = '60'
load_mode = 'AUTO'
# The code will download and load the data automatically as long as the parameters above are set correctly.
viewer = eiscat.EISCATViewer(
      dt_fr, dt_to, site=site, antenna=antenna, modulation=modulation, load_mode='AUTO'
)
viewer.quicklook()
# viewer.save_figure() # comment this if you need to run the following codes
# viewer.show()   # comment this if you need to run the following codes.

"""
The viewer is an instance of the class EISCATViewer, which is a heritage of the class Datahub.
Thus, the variables can be retrieved in the same ways as shown in Example 1. 
"""
n_e = viewer.get_variable('n_e')

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
viewer.save_figure()
# show on screen
viewer.show()
