import geospacelab.datahub as datahub
import madrigal_eiscat_loader as loader
from madrigal_eiscat_variable_config import items as var_config

import geospacelab.toolbox.utilities.pyclass as pyclass
import geospacelab.toolbox.utilities.pybasic as pybasic


default_attrs_for_output = [
    'dt_fr', 'dt_to', 'database', 'facility', 'facility_type',
    'site', 'site_location', 'instrument',
    'experiment', 'pulse_code', 'scan_mode'
]
default_attrs_for_label = [
    'database', 'facility', 'site', 'instrument', 'experiment'
]
default_attrs_for_loader = [
    'dt_fr', 'dt_to', 'input_file_mode', 'input_file_paths', 'input_file_names', 'input_file_num',
    'site', 'instrument', 'experiment', 'scan_mode'
]
default_file_type = 'eiscat-hdf5'


class Dataset(datahub.DatasetBase):
    database = 'Madrigal'
    facility = 'EISCAT'
    facility_type = 'Ground-based incoherent scatter radar'

    def __init__(self, **kwargs):

        self.site = '',
        self.site_location = '',
        self.instrument = ''
        self.experiment = '',
        self.pulse_code = ''
        self.scan_mode = ''
        self.file_type = ''
        kwargs.setdefault('file_type', default_file_type)
        kwargs.setdefault('attrs_for_output', default_attrs_for_output)
        kwargs.setdefault('attrs_for_label', default_attrs_for_label)
        kwargs.setdefault('loader', loader)
        kwargs.setdefault('variable_config', var_config)
        kwargs.setdefault('attrs_for_loader', default_attrs_for_loader)
        super().__init__(**kwargs)
        self.config(**kwargs)

    def _validate_attributes(self):
        pass











