import datetime
import geospacelab.express.omni_viewer as omni

dt_fr = datetime.datetime.strptime('20160314' + '0600', '%Y%m%d%H%M')
dt_to = datetime.datetime.strptime('20160320' + '0600', '%Y%m%d%H%M')

omni_type = 'OMNI2'
omni_res = '1min'
load_mode = 'AUTO'
viewer = omni.OMNIViewer(
    dt_fr, dt_to, omni_type=omni_type, omni_res=omni_res, load_mode=load_mode
)
viewer.quicklook()

# data can be retrieved in the same way as in Example 2:
viewer.list_assigned_variables()
B_x_gsm = viewer.get_variable('B_x_GSM')
# save figure
viewer.save_figure()
# show on screen
viewer.show()