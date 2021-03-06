import numpy as np
import pathlib

from geospacelab.datahub.__init_variable import VariableModel
import geospacelab.config.preferences as pref

import geospacelab.toolbox.utilities.pyclass as pyclass
import geospacelab.toolbox.utilities.pybasic as pybasic
import geospacelab.toolbox.utilities.pylogging as mylog


class DatasetModel(object):
    """ Template for the Dateset class

    """

    def __init__(self, **kwargs):
        self._variables = {}
        self.dt_fr = None
        self.dt_to = None

        self.load_func = 'default'
        self.load_mode = 'AUTO'  # ['AUTO'], 'dialog', 'assigned'
        self.data_root_dir = pref.datahub_data_root_dir
        self.data_file_path_list = []
        self.data_file_patterns = []
        self.data_file_num = 0
        self.data_file_ext = '*'

        self.label_fields = []

    def search_data_files(self, **kwargs):
        done = False
        initial_file_dir = kwargs.pop('initial_file_dir', self.data_root_dir)
        search_pattern = kwargs.pop('search_pattern', '*')
        if str(self.data_file_ext):
            search_pattern = search_pattern + '.' + self.data_file_ext
        files = initial_file_dir.glob(search_pattern)
        if len(files) == 1:
            done = True
            self.data_file_path_list.append(files[0])
        elif len(files) == 0:
            mylog.StreamLogger.warning("Multiple files match!")
            print(files)
        else:
            print('Cannot find the requested data file in {}'.format(initial_file_dir))
        return done

    def open_dialog(self, **kwargs):
        initial_file_dir = kwargs.pop('initial_file_dir', self.data_root_dir)
        title = kwargs.pop('title', 'Open a file:')

        if initial_file_dir is None:
            initial_file_dir = self.data_root_dir

        import tkinter as tk
        from tkinter import simpledialog
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()

        if self.data_file_num == 0:
            self.data_file_num = simpledialog.askinteger('Input dialog', 'Input the number of files:', initialvalue=1)

        for nf in range(self.data_file_num):
            file_types = (('eiscat files', '*.' + self.data_file_ext), ('all files', '*.*'))
            file_name = filedialog.askopenfilename(
                title=title,
                initialdir=initial_file_dir
            )
            self.data_file_path_list.append(pathlib.Path(file_name))

    def check_data_files(self):
        if self.load_mode == 'AUTO':
            self.search_data_files()
        elif self.load_mode == 'dialog':
            self.open_dialog()
        elif self.load_mode == 'assigned':
            if not list(self.data_file_path_list):
                raise ValueError
        else:
            raise AttributeError

    @staticmethod
    def _set_default_attrs(kwargs: dict, default_attrs: dict):
        for key, value in default_attrs:
            kwargs.setdefault(key, value)
        return kwargs

    def label(self, fields=None, separator='_', lowercase=True):
        if fields is None:
            fields = self.label_fields
        sublabels = []
        for attr_name in fields:
            if not str(attr_name):
                sublabels.append('*')
            else:
                sublabels.append(attr_name)
        label = pybasic.str_join(sublabels, separator=separator, lowercase=lowercase)
        return label

    def config(self, logging=True, **kwargs):
        pyclass.set_object_attributes(self, append=False, logging=logging, **kwargs)

    def add_attr(self, logging=True, **kwargs):
        pyclass.set_object_attributes(self, append=True, logging=logging, **kwargs)

    def attrs_to_dict(self, attr_names=None):
        if attr_names is None:
            attr_names = []
        items = {}
        for attr_name in attr_names:
            items[attr_name] = getattr(self, attr_name)
        return items

    def __setitem__(self, key, value):
        if not issubclass(value, VariableModel):
            raise TypeError
        self._variables[key] = value

    def __getitem__(self, key):
        return self._variables[key]

    def __delitem__(self, key):
        del self._variables[key]
        pass

    def add_variable(self, variable, name=None):
        if issubclass(variable, VariableModel):
            pass
        else:
            variable = VariableModel(value=variable, name=name)
        self[variable.name] = variable

    def remove_variable(self, name):
        del self[name]

    def get_variable_names(self):
        return list(self._variables.keys())

    def set_variable(self, **kwargs):
        var_name = kwargs.pop('var_name', '')
        var_config = kwargs.pop('var_config', {})
        if dict(var_config):
            var_config.setdefault('name', var_name)
        else:
            var_configs = kwargs.pop('var_config_items', {})
            var_config = var_configs[var_name]
        var = VariableModel(**var_config)
        var.dataset = self

    def _set_default_variables(self, default_variable_names):
        for var_name in default_variable_names:
            self[var_name] = None



class LoaderBase(object):

    def __init__(self):
        self.dt_fr = None
        self.dt_to = None
        self.file_paths = None
        self.file_names = None
        self.file_num = 0
        self.mode = 'AUTO'
        self.save_pickle = False
        self.load_pickle = False
        self.download = False

    def config(self, logging=True, **kwargs):
        pyclass.set_object_attributes(self, append=False, logging=logging, **kwargs)

    def add_attr(self, logging=True, **kwargs):
        pyclass.set_object_attributes(self, append=True, logging=logging, **kwargs)
#
#
# # BaseClass with the "set_attr" method
# class BaseClass(object):
#     def __init__(self, **kwargs):
#         self.name = kwargs.pop('name', None)
#         self.category = kwargs.pop('kind', None)
#         self.label = None
#
#     def label(self, fields=None, fields_ignore=None, separator='_', lowercase=True):
#         if fields_ignore is None:
#             fields_ignore = ['kind', 'note']
#         sublabels = []
#         if fields is None:
#             attrs = myclass.get_object_attributes(self)
#             for attr, value in attrs.items():
#                 if attr in fields_ignore:
#                     continue
#                 if not isinstance(attr, str):
#                     continue
#                 sublabels.append(value)
#         else:
#             for field in fields:
#                 sublabels.append(getattr(self, field))
#         label = mybasic.str_join(sublabels, separator=separator, lowercase=lowercase)
#         return label
#
#     def set_attr(self, **kwargs):
#         append = kwargs.pop('append', True)
#         logging = kwargs.pop('logging', False)
#         myclass.set_object_attributes(self, append=append, logging=logging, **kwargs)

#
# # Class Database
# class Database(BaseClass):
#     def __init__(self, name='temporary', kind='local', **kwargs):
#
#         super().__init__(name=name, kind=kind)
#         self.set_attr(logging=False, **kwargs)
#
#     def __str__(self):
#         return self.label()
#
#
# # Class Facility
# class Facility(BaseClass):
#     def __init__(self, name=None, category=None, **kwargs):
#         super().__init__(name=name, category=category)
#         self.set_attr(logging=False, **kwargs)
#
#     def __str__(self):
#         return self.label()
#
#
# # Class Instrument
# class Instrument(BaseClass):
#     def __init__(self, name=None, category=None, **kwargs):
#         super().__init__(name=name, category=category)
#         self.set_attr(logging=False, **kwargs)
#
#     def __str__(self):
#         return self.label()
#
#
# # Class Experiment
# class Experiment(BaseClass):
#     def __init__(self, name=None, category=None, **kwargs):
#         super().__init__(name=name, category=category)
#         self.set_attr(logging=False, **kwargs)
#
#     def __str__(self):
#         return self.label()
#
#
# # create the Dataset class
# class Dataset(object):
#     def __init__(self, **kwargs):
#         self.dataDir_root = kwargs.pop('dataDir_root', None)
#         self.dt_fr = kwargs.pop('dt_fr', None)
#         self.dt_to = kwargs.pop('dt_to', None)
#         database_config = kwargs.pop('database_config', {})
#         self.database = kwargs.pop('sources', Database(**database_config))
#         facility_config = kwargs.pop('facility_config', {})
#         self.facility = kwargs.pop('facility', Facility(**facility_config))
#         instrument_config = kwargs.pop('instrument_config', {})
#         self.instrument = kwargs.pop('instrument_opt', Instrument(**instrument_config))
#         experiment_config = kwargs.pop('experiment_config', {})
#         self.experiment = kwargs.pop('experiment', Experiment(**experiment_config))
#         self._variables = {}
#
#     def __getitem__(self, key):
#         return self._variables[key]
#
#     def __setitem__(self, key, value, var_class=Variable):
#         if issubclass(value.__class__, var_class):
#             self._variables[key] = value
#         elif isinstance(value, np.ndarray):
#             self._variables[key] = var_class(name=key, database=self)
#         else:
#             raise TypeError("Variable must be an instance of Geospace Variable or numpy ndarray!")
#
#     def add_variable(self, arr, **kwargs):
#
#         var_name = kwargs.get('name')
#         variable = Variable(arr, **kwargs)
#
#         self.__setitem__(var_name, variable)
#
#     def assign_data(self, Loader, loader_config=None):
#         load_obj = Loader(**loader_config)
#         self.variables
#
#     def label(self):
#         pass
#
#     def __str__(self):
#         return self.label()