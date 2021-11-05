# Licensed under the BSD 3-Clause License
# Copyright (C) 2021 GeospaceLab (geospacelab)
# Author: Lei Cai, Space Physics and Astronomy, University of Oulu

__author__ = "Lei Cai"
__copyright__ = "Copyright 2021, GeospaceLab"
__license__ = "BSD-3-Clause License"
__email__ = "lei.cai@oulu.fi"
__docformat__ = "reStructureText"

import pathlib

from geospacelab.datahub.variable_base import *
from geospacelab import preferences as pref

import geospacelab.toolbox.utilities.pyclass as pyclass
import geospacelab.toolbox.utilities.pybasic as pybasic
import geospacelab.toolbox.utilities.pylogging as mylog


class DatasetModel(object):

    def __init__(self, **kwargs):
        self._variables = {}
        self.name = kwargs.pop('name', None)
        self.kind = kwargs.pop('kind', None)
        self.dt_fr = kwargs.pop('dt_fr', None)
        self.dt_to = kwargs.pop('dt_to', None)

        self.loader = kwargs.pop('loader', None)
        self.downloader = kwargs.pop('downloader', None)
        self.load_mode = kwargs.pop('load_mode', 'AUTO')  # ['AUTO'], 'dialog', 'assigned'
        self.data_root_dir = kwargs.pop('data_root_dir', pref.datahub_data_root_dir)
        self.data_file_paths = kwargs.pop('data_file_paths', [])
        self.data_file_num = kwargs.pop('data_file_num', 0)
        self.data_file_ext = kwargs.pop('data_file_ext', '*')
        self.data_search_recursive = kwargs.pop('data_search_recursive', False)
        self.visual = kwargs.pop('visual', 'off')
        self.time_clip = kwargs.pop('time_clip', True)

        self.label_fields = kwargs.pop('label_fields', [])

    def search_data_files(self, **kwargs):
        done = False
        initial_file_dir = kwargs.pop('initial_file_dir', self.data_root_dir)
        search_pattern = kwargs.pop('search_pattern', '*')
        recursive = kwargs.pop('recursive', self.data_search_recursive)
        allow_multiple_files = kwargs.pop('allow_multiple_files', False)
        if str(self.data_file_ext):
            search_pattern = search_pattern + '.' + self.data_file_ext
        if recursive:
            search_pattern = '**/' + search_pattern
        paths = list(initial_file_dir.glob(search_pattern))

        import natsort
        paths = natsort.natsorted(paths, reverse=False)
        if len(paths) == 1:
            done = True
            self.data_file_paths.append(paths[0])
        elif len(paths) > 1:
            if allow_multiple_files:
                done = True
                self.data_file_paths.extend(paths)
            else:
                mylog.StreamLogger.error("Multiple files found! Restrict the search condition.")
                print(paths)
                raise FileExistsError
        else:
            print('Cannot find the requested data file in {}'.format(initial_file_dir))

        return done

    def open_dialog(self, **kwargs):
        initial_file_dir = kwargs.pop('initial_file_dir', self.data_root_dir)
        title = kwargs.pop('title', 'Open a file:')

        if initial_file_dir is None:
            initial_file_dir = self.data_root_dir

        self.data_file_num = kwargs.pop('data_file_num', self.data_file_num)
        if self.data_file_num == 0:
            value = input("How many files will be loaded? Input the number: ")
            self.data_file_num = int(value)
        for nf in range(self.data_file_num):
            value = input("Input the No. {} file's full path: ".format(str(nf)))
            fp = pathlib.Path(value)
            if not fp.is_file():
                mylog.StreamLogger.warning("The input file does not exist!")
                return
            self.data_file_paths.append(fp)

        # import tkinter as tk
        # from tkinter import simpledialog
        # from tkinter import filedialog
        #
        # root = tk.Tk()
        # root.withdraw()
        # self.data_file_num = kwargs.pop('data_file_num', self.data_file_num)
        # if self.data_file_num == 0:
        #     self.data_file_num = simpledialog.askinteger('Input dialog', 'Input the number of files:', initialvalue=1)
        #
        # for nf in range(self.data_file_num):
        #     file_types = (('eiscat files', '*.' + self.data_file_ext), ('all files', '*.*'))
        #     file_name = filedialog.askopenfilename(
        #         title=title,
        #         initialdir=initial_file_dir
        #     )
        #     self.data_file_paths.append(pathlib.Path(file_name))

    def check_data_files(self, **kwargs):
        self.load_mode = kwargs.pop('load_mode', self.load_mode)
        if self.load_mode == 'AUTO':
            self.search_data_files(**kwargs)
        elif self.load_mode == 'dialog':
            self.open_dialog(**kwargs)
        elif self.load_mode == 'assigned':
            self.data_file_paths = kwargs.pop('data_file_paths', self.data_file_paths)
            if not list(self.data_file_paths):
                raise ValueError
        else:
            raise AttributeError
        self.data_file_num = len(self.data_file_paths)

    def time_filter_by_range(self, var_datetime=None, var_datetime_name=None):
        if var_datetime is None and var_datetime_name is None:
            var_datetime = self['DATETIME']
        if var_datetime_name is not None:
            var_datetime = self[var_datetime_name]
        if var_datetime.value is None:
            return
        inds = np.where((var_datetime.value.flatten() >= self.dt_fr) & (var_datetime.value.flatten() <= self.dt_to))[0]
        self.time_filter_by_inds(inds, var_datetime=var_datetime)

    def time_filter_by_inds(self, inds, var_datetime=None):
        if inds is None:
            return
        if not list(inds):
            return
        if var_datetime is None:
            var_datetime = self['DATETIME']

        shape_0 = var_datetime.value.shape[0]
        for var in self._variables.values():
            if var.value.shape[0] == shape_0 and len(var.value.shape) > 1:
                var.value = var.value[inds, ::]

    def label(self, fields=None, separator=' | ', lowercase=True):
        if fields is None:
            fields = self.label_fields
        sublabels = []
        for attr_name in fields:
            if not str(attr_name):
                sublabels.append('*')
            else:
                sublabels.append(getattr(self, attr_name))
        label = pybasic.str_join(*sublabels, separator=separator, lowercase=lowercase)
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

    def list_all_variables(self):
        # print header
        label = self.label()
        mylog.simpleinfo.info("Dataset: {}".format(label))
        mylog.simpleinfo.info("Printing all of the variables ...")
        mylog.simpleinfo.info('{:^20s}{:^30s}'.format('No.', 'Variable name'))
        for ind, var_name in enumerate(self._variables.keys()):
            mylog.simpleinfo.info('{:^20d}{:30s}'.format(ind+1, var_name))
        print()

    def keys(self):
        return self._variables.keys()

    def __setitem__(self, key, value):
        if value is not None:
            if not issubclass(value.__class__, VariableModel):
                raise TypeError
        self._variables[key] = value

    def __getitem__(self, key):
        # if key not in self._variables.keys():
        #    raise KeyError
        return self._variables[key]

    def __delitem__(self, key):
        del self._variables[key]
        pass

    def __repr__(self):
        rep = f"GeospaceLab Dataset object <{self.label()}, " + \
              f"starting time: {str(self.dt_fr)}, stopping time: {str(self.dt_to)}>"
        return rep.replace('geospacelab.datahub.sources.', '')

    # def add_variable(self, variable, name=None):
    #     if issubclass(variable, VariableModel):
    #         pass
    #     else:
    #         variable = VariableModel(value=variable, name=name, visual=self.visual)
    #     self[variable.name] = variable
    def exist(self, var_name):
        if var_name in self._variables.keys():
            return True
        else:
            return False

    def remove_variable(self, name):
        del self[name]

    def get_variable_names(self):
        return list(self._variables.keys())

    def add_variable(self, var_name, configured_variables=None, **kwargs):
        if configured_variables is None:
            configured_variables = {}
        if var_name in configured_variables.keys():
            configured_variable = configured_variables[var_name]
            if type(configured_variable) is dict:
                self[var_name] = VariableModel(**configured_variable)
            elif issubclass(configured_variable.__class__, VariableModel):
                self[var_name] = configured_variable.clone()
            self[var_name].dataset = self
            self[var_name].visual = self.visual
        else:
            self[var_name] = VariableModel(dataset=self, visual=self.visual, **kwargs)
        return self[var_name]

    def _set_default_variables(self, default_variable_names, configured_variables=None):
        if configured_variables is None:
            configured_variables = {}
        for var_name in default_variable_names:
            self.add_variable(var_name, configured_variables=configured_variables)

    @property
    def data_root_dir(self):
        return self._data_root_dir

    @data_root_dir.setter
    def data_root_dir(self, path):
        self._data_root_dir = pathlib.Path(path)


class DownloaderModel(object):
    """
    Downloader Model for downloading data files from different data sources

    Parameters
    ===========
    :param dt_fr: starting time.
    :type dt_fr: datetime.datetime
    :param dt_to: stopping time.
    :type dt_to: datetime.datetime
    :param data_file_root_dir: the root directory storing the data.
    :type data_file_root_dir: str or pathlib.Path object
    :param done: if the files are downloaded or not [False].
    :type done: bool
    :param force: Force the downloader to download the data files, [True].
    :type force: bool
    :param direct_download: Download the data directly without calling the function download [True].
    :type direct_download: bool
    """

    def __init__(self, dt_fr, dt_to, data_file_root_dir=None, force=True, direct_download=True, **kwargs):

        self.dt_fr = dt_fr
        self.dt_to = dt_to
        self.data_file_root_dir = data_file_root_dir

        self.files = []

        self.force = force
        self.direct_download = direct_download
        self.done = False

        if self.direct_download:
            self.done = self.download(**kwargs)

    def download(self, **kwargs):
        """
        Download the data from a http or ftp server. Implementation must be added according to different data sources
        :param kwargs: keywords for downloading the data files.
        """
        raise NotImplemented

# class LoaderBase(object):
#
#     def __init__(self):
#         self.dt_fr = None
#         self.dt_to = None
#         self.file_paths = None
#         self.file_names = None
#         self.file_num = 0
#         self.mode = 'AUTO'
#         self.save_pickle = False
#         self.load_pickle = False
#         self.download = False
#
#     def config(self, logging=True, **kwargs):
#         pyclass.set_object_attributes(self, append=False, logging=logging, **kwargs)
#
#     def add_attr(self, logging=True, **kwargs):
#         pyclass.set_object_attributes(self, append=True, logging=logging, **kwargs)
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