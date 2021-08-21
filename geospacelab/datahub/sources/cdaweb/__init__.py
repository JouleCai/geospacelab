# Licensed under the BSD 3-Clause License
# Copyright (C) 2021 GeospaceLab (geospacelab)
# Author: Lei Cai, Space Physics and Astronomy, University of Oulu

__author__ = "Lei Cai"
__copyright__ = "Copyright 2021, GeospaceLab"
__license__ = "BSD-3-Clause License"
__email__ = "lei.cai@oulu.fi"
__docformat__ = "reStructureText"


from geospacelab.datahub import DatabaseModel


class CDAWebDatabase(DatabaseModel):
    def __new__(cls, str_in, **kwargs):
        obj = super().__new__(cls, str_in, **kwargs)
        return obj


cdaweb_database = CDAWebDatabase('CDAWeb')
cdaweb_database.url = 'https://cdaweb.gsfc.nasa.gov/index.html/'
cdaweb_database.category = 'online database'