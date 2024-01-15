from geospacelab.config import pref
import geospacelab.datahub.sources.madrigal.utilities as utilities

from geospacelab.datahub import DatabaseModel


class WDCDatabase(DatabaseModel):
    def __new__(cls, str_in, **kwargs):
        obj = super().__new__(cls, str_in, **kwargs)
        return obj


wdc_database = WDCDatabase('WDC')
wdc_database.url = 'http://wdc.kugi.kyoto-u.ac.jp/'
wdc_database.category = 'online database'
wdc_database.Notes = '''
- Data Usage Rules
The rules for the data use and exchange are defined by the International Council for Science - World Data System (ICSU-WDS) Data Sharing Principles. 
The data and services at the WDC Kyoto are available for scientific use without restrictions, 
but for the real-time (quicklook) data, please contact our staff (wdc-service@kugi.kyoto-u.ac.jp) before using those in publications and presentations. 
The WDC Kyoto does not allow commercial applications of the geomagnetic indices.
'''

try:
    default_user_email = pref.user_config['datahub']['wdc']['user_email']
except KeyError:
    if pref._on_rtd:
        default_user_email = 'geospacelab@gmail.com'
        save = 'y'
    else:
        print("Inputs for accessing the WDC (wdc.kugi.kyoto-u.ac.jp) database.")
        default_user_email = input("User's email: ")
        save = input("Save as default? [y]/n: ")
    if save.lower() in ['', 'y', 'yes']:
        uc = pref.user_config
        uc.setdefault('datahub', {})
        uc['datahub'].setdefault('wdc', {})
        uc['datahub']['wdc']['user_email'] = default_user_email
        pref.set_user_config(user_config=uc, set_as_default=True)
