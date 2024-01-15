from geospacelab.config import pref
import geospacelab.datahub.sources.madrigal.utilities as utilities

from geospacelab.datahub import DatabaseModel


class GFZDatabase(DatabaseModel):
    def __new__(cls, str_in, **kwargs):
        obj = super().__new__(cls, str_in, **kwargs)
        return obj


gfz_database = GFZDatabase('WDC')
gfz_database.url = 'https://www.gfz-potsdam.de/en/section/geomagnetism/data-products-services/'
gfz_database.category = 'online database'
gfz_database.Notes = ''
