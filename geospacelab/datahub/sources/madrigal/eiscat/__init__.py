import geospacelab.datahub as datahub
import geospacelab.datahub.sources.madrigal.eiscat.load_madrigal_eiscat as loader
import geospacelab.datahub.sources.madrigal.eiscat.config_variables_madrigal_eiscat as config_variables

import geospacelab.toolbox.utilities.pyclass as pyclass
import geospacelab.toolbox.utilities.pybasic as pybasic

default_attributes = {
    'database': 'Madrigal',
    'facility': 'eiscat',
    'category': 'incoherent scatter radar',
    'file_type': 'eiscat-hdf5',
    'file_dir': '',
    'site_name': '',
    'site_fullname': '',
    'site_location': '',
    'site_antenna': '',
    'exp_name': '',
    'exp_pulse_code': '',
    'exp_scan_mode': '',
}

default_label_fields = ['database', 'facility', 'site_name', 'site_antenna', 'exp_name']


class Dataset(datahub.DatasetBase):
    def __init__(self, **kwargs):
        kwargs.setdefault('default_attributes', default_attributes)
        kwargs.setdefault('default_label_attribute_names', default_label_fields)
        super().__init__(**kwargs)

    def _validate_attributes(self):
        pass

    def assign_data(self):
        config_load_data = {
            'file_type': self.file_type,
            ''
        }
        load_obj = loader.load_data(**kwargs)
        for var_name in config_variables.items.keys():
            variable = datahub.Variable(load_obj[var_name], **config_variables.items[var_name])
            self.add_variable(variable)





