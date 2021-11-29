import datetime
import geospacelab.express.eiscat_dashboard as eiscat

dt_fr = datetime.datetime.strptime('20201209' + '1800', '%Y%m%d%H%M')
dt_to = datetime.datetime.strptime('20201210' + '0600', '%Y%m%d%H%M')

# check the eiscat-hdf5 filename from the EISCAT schedule page, e.g., "EISCAT_2020-12-10_beata_60@uhfa.hdf5"
site = 'UHF'
antenna = 'UHF'
modulation = '60'
load_mode = 'AUTO'
# The code will download and load the data automatically as long as the parameters above are set correctly.
viewer = eiscat.EISCATDashboard(
      dt_fr, dt_to, site=site, antenna=antenna, modulation=modulation, load_mode='AUTO'
)
viewer.quicklook()