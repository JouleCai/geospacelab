# Licensed under the BSD 3-Clause License
# Copyright (C) 2021 GeospaceLab (geospacelab)
# Author: Lei Cai, Space Physics and Astronomy, University of Oulu

__author__ = "Lei Cai"
__copyright__ = "Copyright 2021, GeospaceLab"
__license__ = "BSD-3-Clause License"
__email__ = "lei.cai@oulu.fi"
__docformat__ = "reStructureText"


from geospacelab.config._preferences import Preferences

prf = pref = Preferences()

try:
    opt = pref.user_config['visualization']
except KeyError:
    uc = pref.user_config
    uc['visualization'] = dict()
    uc['visualization']['mpl'] = dict()
    pref.set_user_config(user_config=uc, set_as_default=True)
