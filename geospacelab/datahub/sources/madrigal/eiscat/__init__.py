import geospacelab.datahub as datahub
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

    def load_data(self, **kwargs):
        loader = self._loader_class
