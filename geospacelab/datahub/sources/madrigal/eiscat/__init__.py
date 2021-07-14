import geospacelab.datahub as datahub
import madrigal_eiscat_loader as loader
import madrigal_eiscat_config as config

import geospacelab.toolbox.utilities.pyclass as pyclass
import geospacelab.toolbox.utilities.pybasic as pybasic



default_attrs_to_file = ['dt_fr', 'dt_to', 'database', 'facility', 'facility_type',
                         'site', 'site_location', 'instrument',
                         'experiment', 'pulse_code', 'scan_mode']
default_attrs_labeled = ['database', 'facility', 'site', 'instrument', 'experiment']


class Dataset(datahub.DatasetBase):
    database = 'Madrigal'
    facility = 'EISCAT'
    facility_type = 'Ground-based incoherent scatter radar'
    site = '',
    site_location = '',
    instrument = ''
    experiment = '',
    pulse_code = ''
    scan_mode = ''
    attrs_to_file = []
    attrs_labeled = []
    file_type = 'eiscat-hdf5'

    def __init__(self, **kwargs):
        kwargs.setdefault('attrs_to_file', default_attrs_to_file)
        kwargs.setdefault('label_fields', default_attrs_labeled)
        self.config(**kwargs)

        super().__init__(**kwargs)

    def _validate_attributes(self):
        pass

    def assign_variable(self, var_name):
        var_config = config.items[var_name]
        datahub.VariableModel(self._variables[var_name], **var_config)

    def load_data(self):
        load_config = self.attrs_to_dict(
            ['dt_fr', 'dt_to', 'input_file_mode', 'input_file_paths', 'input_file_names', 'input_file_num',
             'site', 'instrument', 'experiment', 'scan_mode']
        )
        if self._loader is None:
            load_obj = loader.Loader(**load_config)
        else:
            load_obj = self._loader.Loader(**load_config)







