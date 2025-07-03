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
import geospacelab.toolbox.utilities.pylogging as mylog


class AMPEREDatabase(DatabaseModel):
    def __new__(cls, str_in, **kwargs):
        obj = super().__new__(cls, str_in, **kwargs)
        return obj


ampere_database = AMPEREDatabase('AMPERE')
ampere_database.url = 'https://ampere.jhuapl.edu/'
ampere_database.category = 'online database'


try:
    default_user_name = pref.user_config['datahub']['ampere']['user_name']
except KeyError:
    print("Inputs for accessing the JHUAPL/AMPERE database.")
    if pref._on_rtd:
        default_user_name = 'GeospaceLAB'
        save = 'y'
    else:
        mylog.StreamLogger.info("A username is needed to download the AMPERE data. If you don't have one, register as a new user on the webpage: https://ampere.jhuapl.edu/browse/!")
        default_user_name = input("Set the username: ")
        save = input("Save as default? [y]/n: ")
    if save.lower() in ['', 'y', 'yes']:
        uc = pref.user_config
        uc.setdefault('datahub', {})
        uc['datahub'].setdefault('ampere', {})
        uc['datahub']['ampere']['user_name'] = default_user_name
        pref.set_user_config(user_config=uc, set_as_default=True)
