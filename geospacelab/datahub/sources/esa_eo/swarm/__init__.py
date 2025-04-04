# Licensed under the BSD 3-Clause License
# Copyright (C) 2021 GeospaceLab (geospacelab)
# Author: Lei Cai, Space Physics and Astronomy, University of Oulu
import time

import keyring
import getpass


from geospacelab.config import pref
import geospacelab.toolbox.utilities.pylogging as mylog

from geospacelab.datahub import FacilityModel


class SWARMFacility(FacilityModel):
    def __new__(cls, str_in, **kwargs):
        obj = super().__new__(cls, str_in, **kwargs)
        return obj


swarm_facility = SWARMFacility('SWARM')
swarm_facility.url = 'https://earth.esa.int/eogateway/missions/swarm/data'
swarm_facility.category = 'SWARM online database'
swarm_facility.Notes = ''

from_keyring = False
try:
    default_username = pref.user_config['datahub']['swarm']['username']
    from_keyring = True
except KeyError:
    print("Initialization for accessing the Swarm DISC database.")
    if pref._on_rtd:
        default_username = 'GeospaceLab'
        save = 'y'
    else:
        mylog.StreamLogger.info(
            """
            Since 2025, it requires logining to Swarm DISC FTP service with your EO's account.
            More information can be found here: https://earth.esa.int/eogateway/faq/how-do-i-access-swarm-data.
            GeospaceLAB use keyring (https://github.com/jaraco/keyring) to store your acount name and password.
            If you don't want to store the password, select "n" for "Save as default".
            And you may reset your password using keyring: keyring.set_password('ESA Earth Online', YOUR_USERNAME, YOUR_NEW_PASSWORD)
            """
            )
        time.sleep(0.5)
        default_username = input("EO username: ")
        save = input("Save as default? [y]/n: ")
    if save.lower() in ['', 'y', 'yes']:
        uc = pref.user_config
        uc.setdefault('datahub', {})
        uc['datahub'].setdefault('swarm', {})
        uc['datahub']['swarm']['username'] = default_username
        pref.set_user_config(user_config=uc, set_as_default=True)
        from_keyring = True
    else:
        eo_password = getpass.getpass("EO password: ")

if from_keyring:
    eo_password = keyring.get_password('ESA Earth Online', default_username)

if pref._on_rtd:
    eo_password = ''

if eo_password is None:
    eo_password = getpass.getpass("EO password: ")
    keyring.set_password('ESA Earth Online', default_username, eo_password)


