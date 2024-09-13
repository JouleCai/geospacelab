# Licensed under the BSD 3-Clause License
# Copyright (C) 2021 GeospaceLab (geospacelab)
# Author: Lei Cai, Space Physics and Astronomy, University of Oulu

from geospacelab.datahub import DatabaseModel


class SuperDARNDatabase(DatabaseModel):
    def __new__(cls, str_in, **kwargs):
        obj = super().__new__(cls, str_in, **kwargs)
        return obj


superdarn_database = SuperDARNDatabase('SuperDARN')
superdarn_database.url = 'http://vt.superdarn.org/tiki-index.php'
superdarn_database.category = 'online database'
superdarn_database.Notes = ''
