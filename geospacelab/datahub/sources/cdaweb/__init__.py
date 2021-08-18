from geospacelab.datahub import DatabaseModel


class CDAWebDatabase(DatabaseModel):
    def __new__(cls, str_in, **kwargs):
        obj = super().__new__(cls, str_in, **kwargs)
        return obj


cdaweb_database = CDAWebDatabase('CDAWeb')
cdaweb_database.url = 'https://cdaweb.gsfc.nasa.gov/index.html/'
cdaweb_database.category = 'online database'