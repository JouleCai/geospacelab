# Licensed under the BSD 3-Clause License
# Copyright (C) 2021 GeospaceLAB (geospacelab)
# Author: Lei Cai, Space Physics and Astronomy, University of Oulu

__author__ = "Lei Cai"
__copyright__ = "Copyright 2021, GeospaceLAB"
__license__ = "BSD-3-Clause License"
__email__ = "lei.cai@oulu.fi"

import geospacelab.datahub.sources.madrigal.utilities as utilities

from geospacelab.datahub import DatabaseModel


class NIPRDatabase(DatabaseModel):
    def __new__(cls, str_in, **kwargs):
        obj = super().__new__(cls, str_in, **kwargs)
        return obj


nipr_database = NIPRDatabase('NIPR')
nipr_database.url = 'http://polaris.nipr.ac.jp/~asi-dp/watcam/'
nipr_database.category = 'online database'
nipr_database.Notes = ''
