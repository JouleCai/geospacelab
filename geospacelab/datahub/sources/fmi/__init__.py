# Licensed under the BSD 3-Clause License
# Copyright (C) 2021 GeospaceLAB (geospacelab)
# Author: Lei Cai, Space Physics and Astronomy, University of Oulu

__author__ = "Lei Cai"
__copyright__ = "Copyright 2021, GeospaceLab"
__license__ = "BSD-3-Clause License"
__email__ = "lei.cai@oulu.fi"

import geospacelab.datahub.sources.madrigal.utilities as utilities

from geospacelab.datahub import DatabaseModel


class FMIDatabase(DatabaseModel):
    def __new__(cls, str_in, **kwargs):
        obj = super().__new__(cls, str_in, **kwargs)
        return obj


fmi_database = FMIDatabase('FMI')
fmi_database.url = 'https://space.fmi.fi/'
fmi_database.category = 'online database'
fmi_database.Notes = ''
