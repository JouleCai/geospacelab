# __all__ = ['FacilityModel']

import geospacelab.toolbox.utilities.pyclass as pyclass


class StrAttrBase(str):
    def __new__(cls, str_in):
        if issubclass(str_in.__class__, StrAttrBase):
            obj = str_in
        else:
            obj = str.__new__(cls, str_in)
        return obj

    def config(self, logging=True, **kwargs):
        pyclass.set_object_attributes(self, append=False, logging=logging, **kwargs)


class DatabaseModel(StrAttrBase):
    _category = ''
    _url = ''
    _description = ''
    _notes = []

    def __new__(cls, str_in, **kwargs):
        obj = super().__new__(cls, str_in)
        obj.category = kwargs.pop('category', cls._category)
        obj.url = kwargs.pop('url', cls._url)
        obj.description = kwargs.pop('description', cls._description)
        obj.notes = kwargs.pop('notes', cls._notes)
        return obj


class FacilityModel(StrAttrBase):
    _category = ''
    _url = ''
    _description = ''
    _notes = []

    def __new__(cls, str_in, **kwargs):
        obj = super().__new__(cls, str_in)
        obj.category = kwargs.pop('category', cls._category)
        obj.url = kwargs.pop('url', cls._url)
        obj.description = kwargs.pop('description', cls._description)
        obj.notes = kwargs.pop('notes', cls._notes)
        return obj


class DataProductModel(StrAttrBase):
    _category = ''
    _url = ''
    _description = ''
    _notes = []

    def __new__(cls, str_in, **kwargs):
        obj = super().__new__(cls, str_in)
        obj.category = kwargs.pop('category', cls._category)
        obj.url = kwargs.pop('url', cls._url)
        obj.description = kwargs.pop('description', cls._description)
        obj.notes = kwargs.pop('notes', cls._notes)
        return obj


class InstrumentModel(StrAttrBase):
    _category = ''
    _description = ''
    _notes = []

    def __new__(cls, str_in, **kwargs):
        obj = super().__new__(cls, str_in)
        obj.category = kwargs.pop('category', cls._category)
        obj.description = kwargs.pop('description', cls._description)
        obj.notes = kwargs.pop('notes', cls._notes)
        return obj


class SiteModel(StrAttrBase):
    _category = ''
    _location = {'GEO_LAT': None, 'GEO_LON': None, 'GEO_ALT': None}
    _description = ''
    _notes = []

    def __new__(cls, str_in, **kwargs):
        obj = super().__new__(cls, str_in)
        obj.category = kwargs.pop('category', cls._category)
        obj.location = kwargs.pop('url', cls._location)
        obj.description = kwargs.pop('description', cls._description)
        obj.notes = kwargs.pop('notes', cls._notes)
        return obj


class MetadataModel(object):
    def __init__(self):
        self.database = ''

    @property
    def database(self):
        return self._database

    @database.setter
    def database(self, value):
        self._database = value

    def config(self, logging=True, **kwargs):
        pyclass.set_object_attributes(self, append=False, logging=logging, **kwargs)