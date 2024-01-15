# Licensed under the BSD 3-Clause License
# Copyright (C) 2021 GeospaceLab (geospacelab)
# Author: Lei Cai, Space Physics and Astronomy, University of Oulu


from geospacelab.datahub import DatabaseModel


class JHUAPLDatabase(DatabaseModel):
    def __new__(cls, str_in, **kwargs):
        obj = super().__new__(cls, str_in, **kwargs)
        return obj


jhuapl_database = JHUAPLDatabase('JHUAPL')
jhuapl_database.url = 'https://www.jhuapl.edu/'
jhuapl_database.category = 'online database'
jhuapl_database.Notes = ''
