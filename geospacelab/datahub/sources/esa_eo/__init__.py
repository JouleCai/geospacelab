# Licensed under the BSD 3-Clause License
# Copyright (C) 2021 GeospaceLab (geospacelab)
# Author: Lei Cai, Space Physics and Astronomy, University of Oulu


from geospacelab.datahub import DatabaseModel


class ESAEODatabase(DatabaseModel):
    def __new__(cls, str_in, **kwargs):
        obj = super().__new__(cls, str_in, **kwargs)
        return obj


esaeo_database = ESAEODatabase('ESA/EarthOnline')
esaeo_database.url = 'https://earth.esa.int/'
esaeo_database.category = 'online database'
esaeo_database.Notes = ''