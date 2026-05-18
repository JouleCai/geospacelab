# Licensed under the BSD 3-Clause License
# Copyright (C) 2021 GeospaceLab (geospacelab)
# Author: Lei Cai, Space Physics and Astronomy, University of Oulu

import time
import keyring
import getpass

from geospacelab.config import pref
import geospacelab.toolbox.utilities.pylogging as mylog
from geospacelab.datahub import DatabaseModel


class ESAEODatabase(DatabaseModel):
    def __new__(cls, str_in, **kwargs):
        obj = super().__new__(cls, str_in, **kwargs)
        return obj


esaeo_database = ESAEODatabase('ESA/EarthOnline')
esaeo_database.url = 'https://earth.esa.int/'
esaeo_database.category = 'online database'
esaeo_database.Notes = ''

from_keyring = False
try:
    eo_username = pref.user_config['datahub']['esa_eo']['username']
    from_keyring = True
except KeyError:
    print("Initialization for accessing the ESA Earth Online database.")
    if pref._on_rtd:
        eo_username = 'GeospaceLAB'
        save = 'y'
    else:
        mylog.StreamLogger.info(
            """
            Since 2025, it requires logging in to Swarm Dissemination Server with an ESA EO account.
            Register your account at https://eoiam-idp.eo.esa.int/.
            More information can be found here: https://earth.esa.int/eogateway/faq/how-do-i-access-swarm-data.
            GeospaceLAB use keyring (https://github.com/jaraco/keyring) to store your account name and password.
            If you don't want to store the password, select "n" for "Save as default".
            To reset your password using keyring in a Python script: keyring.set_password('ESA Earth Online', YOUR_USERNAME, YOUR_NEW_PASSWORD)
            """
            )
        time.sleep(0.5)
        eo_username = input("ESA EO username: ")
        save = input("Save as default? [y]/n: ")
    if save.lower() in ['', 'y', 'yes']:
        uc = pref.user_config
        uc.setdefault('datahub', {})
        uc['datahub'].setdefault('esa_eo', {})
        uc['datahub']['esa_eo']['username'] = eo_username
        pref.set_user_config(user_config=uc, set_as_default=True)
        from_keyring = True
        
        eo_password = keyring.get_password('ESA Earth Online', eo_username)
        if eo_password is not None:
            reset = input("Reset password? [y]/n: ")
            if reset.lower() in ['y', 'yes']:
                eo_password = getpass.getpass("ESA EO password: ")
                keyring.set_password('ESA Earth Online', eo_username, eo_password)
    else:
        eo_password = getpass.getpass("ESA EO password: ")

if from_keyring:
    eo_password = keyring.get_password('ESA Earth Online', eo_username)

if pref._on_rtd:
    eo_password = ''

if eo_password is None:
    eo_password = getpass.getpass("ESA EO password: ")
    keyring.set_password('ESA Earth Online', eo_username, eo_password)
