# Licensed under the BSD 3-Clause License
# Copyright (C) 2021 GeospaceLab (geospacelab)
# Author: Lei Cai, Space Physics and Astronomy, University of Oulu

__author__ = "Lei Cai"
__copyright__ = "Copyright 2021, GeospaceLab"
__license__ = "BSD-3-Clause License"
__email__ = "lei.cai@oulu.fi"
__docformat__ = "reStructureText"


from geospacelab.config import pref
import geospacelab.datahub.sources.madrigal.utilities as utilities

from geospacelab.datahub import DatabaseModel


class SuperMAGDatabase(DatabaseModel):
    def __new__(cls, str_in, **kwargs):
        obj = super().__new__(cls, str_in, **kwargs)
        return obj


supermag_database = SuperMAGDatabase('SuperMAG')
supermag_database.url = 'https://supermag.jhuapl.edu/'
supermag_database.category = 'online database'


try:
    default_user_fullname = pref.user_config['datahub']['supermag']['username']
    # default_user_email = preferences.user_config['datahub']['madrigal']['user_email']
    # default_user_affiliation = preferences.user_config['datahub']['madrigal']['user_email']
except KeyError:
    print("Inputs for accessing the SuperMAG database.")
    if pref._on_rtd:
        default_username = 'GeospaceLab'
        # default_user_email = 'geospacelab@gmail.com'
        # default_user_affiliation = 'GeospaceLab'
        save = 'y'
    else:
        default_username = input("Username: ")
        # default_user_email = input("User's email: ")
        # default_user_affiliation = input("User's affiliation: ")
        save = input("Save as default? [y]/n: ")
    if save.lower() in ['', 'y', 'yes']:
        uc = pref.user_config
        uc.setdefault('datahub', {})
        uc['datahub'].setdefault('supermag', {})
        uc['datahub']['supermag']['username'] = default_username
        # uc['datahub']['madrigal']['user_email'] = default_user_email
        # uc['datahub']['madrigal']['user_affiliation'] = default_user_affiliation
        pref.set_user_config(user_config=uc, set_as_default=True)

