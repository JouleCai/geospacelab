"""

The module datahub is the data manager in GeospaceLab. The datahub has three core class-based components:

    - :class:`DataHub <geospacelab.datahub.DataHub>` manages a set of datasets docked or added to the datahub.
    - :class:`Dataset <geospacelab.datahub.DatasetModel>` manages a set of variables loaded from a data source.
    - :class:`Variable <geospacelab.datahub.VariableModel>` records the value, error, and various attributes \
    (e.g., name, label, unit, depends, ndim, ...) of a variable.

"""

# Licensed under the BSD 3-Clause License
# Copyright (C) 2021 GeospaceLab (geospacelab)
# Author: Lei Cai, Space Physics and Astronomy, University of Oulu

__author__ = "Lei Cai"
__copyright__ = "Copyright 2021, GeospaceLab"
__license__ = "BSD-3-Clause License"
__email__ = "lei.cai@oulu.fi"
__docformat__ = "reStructureText"

# __all__ = ['DataHub', 'DatasetModel', 'VariableModel',]

import datetime
import pathlib
import numpy as np

import geospacelab.toolbox.utilities.pylogging as mylog
import geospacelab.toolbox.utilities.pybasic as pybasic
import geospacelab.toolbox.utilities.pyclass as pyclass
from geospacelab.config import pref


class DatasetBase(object):
    """
    A dataset is a dictionary-like object used for downloading and loading data from a data source. The items in
    the dataset are the variables loaded from the data files. The parameters listed below are the general attributes used
    for the dataset class and its inheritances.

    :ivar str name: The name of the dataset.
    :ivar str kind: The type of the dataset. 'sourced': the data source has been added in the package,
      'temporary': a dataset added temporarily, 'user-defined': a data source defined by the user.
    :ivar datetime.datetime or None dt_fr: the starting time of the data records.
    :ivar datetime.datetime or None dt_fr: the starting time of the data records.
    :ivar str, {"on", "off"} visual: If "on", append the Visual object to the Variable object.
    :ivar list label_fields: A list of strings, indicating the fields used for generating the dataset label.
    """

    from geospacelab.datahub.__variable_base__ import VariableBase as VariableModel
    __variable_model__ = VariableModel

    def __init__(self,
                 dt_fr: datetime.datetime = None,
                 dt_to: datetime.datetime = None,
                 name: str = '',
                 kind: str = '',
                 visual: str = 'off',
                 label_fields: list = ('name', 'kind'), **kwargs):
        """
        Initial inputs to create the object.

            :param name: The name of the dataset.
            :type name: str or None
            :param kind: The type of the dataset. 'sourced': the data source has been added in the package,
              'temporary': a dataset added temporarily, 'user-defined': a data source defined by the user.
            :type kind: {'sourced', 'temporary', 'user-defined'}
            :param dt_fr: the starting time of the data records.
            :type dt_fr: datetime.datetime, default: None
            :param dt_to: the stopping time of the the data records.
            :type dt_to: datetime.datetime, default: None
            :param visual: If "on", append the attribute visual to the vairables.
            :type visual: {'on', 'off'}
            :param label_fields: A list of strings, indicating the fields used for generating the dataset label.
            :type label_fields: list, {['name', 'kind']}
        """

        self._variables = {}
        self.name = name
        self.kind = kind
        self.dt_fr = dt_fr
        self.dt_to = dt_to
        self.visual = visual

        self.label_fields = label_fields

    def add_variable(
            self, var_name: str,
            configured_variables=None, configured_variable_name=None,
            variable_class=None, **kwargs) -> __variable_model__:
        """
        Add a variable to the dataset.

        :param var_name:
        :param configured_variables:
        :param variable_class:
        :param kwargs:
        :return:
        """
        if variable_class is None:
            variable_class = self.__variable_model__
        if configured_variables is None:
            configured_variables = {}
        if var_name in configured_variables.keys() or configured_variable_name is not None:
            if configured_variable_name is None:
                vn = var_name
            else:
                vn = configured_variable_name
            configured_variable = configured_variables[vn]
            if type(configured_variable) is dict:
                self[var_name] = variable_class(**configured_variable)
            elif issubclass(configured_variable.__class__, self.__variable_model__):
                self[var_name] = configured_variable.clone()
            # self[var_name].dataset = self
            self[var_name].visual = self.visual
        else:
            self[var_name] = variable_class(dataset=self, name=var_name, visual=self.visual, **kwargs)
        return self[var_name]
    
    def clone_variables(self, ds, variable_names=None):
        if not isinstance(ds, DatasetBase):
            raise TypeError
        if variable_names is None:
            variable_names = ds.keys()
        for var_name in variable_names:
            self[var_name] = ds[var_name].clone()
            self[var_name].dataset = self
        return None

    def label(self, fields=None, separator=' | ', lowercase=True, num_to_str=True) -> str:
        """
        Return a label of the data set.
        :param fields: The attribute names for the label.
        :param separator: A separator between two attributes.
        :param lowercase: Show lowercase letters only.
        :return: label
        """
        if fields is None:
            fields = self.label_fields
        sublabels = []
        for attr_name in fields:
            if not str(attr_name):
                sublabels.append('*')
            else:
                attr = getattr(self, attr_name)
                if pybasic.isnumeric(attr) and num_to_str:
                    attr = str(attr)
                sublabels.append(attr)
        label = pybasic.str_join(*sublabels, separator=separator, lowercase=lowercase)
        return label

    def config(self, logging: bool = True, **kwargs) -> None:
        """
        Configure the attributes of the dataset.

        :param logging: Show logging if True.
        :param kwargs:
        :return:
        """
        pyclass.set_object_attributes(self, append=False, logging=logging, **kwargs)

    def add_attr(self, logging=True, **kwargs) -> None:
        pyclass.set_object_attributes(self, append=True, logging=logging, **kwargs)

    def attrs_to_dict(self, attr_names=None) -> dict:
        if attr_names is None:
            attr_names = []
        items = {}
        for attr_name in attr_names:
            items[attr_name] = getattr(self, attr_name)
        return items

    def list_all_variables(self) -> None:
        # print header
        label = self.label()
        mylog.simpleinfo.info("Dataset: {}".format(label))
        mylog.simpleinfo.info("Printing all of the variables ...")
        mylog.simpleinfo.info('{:^20s}{:^30s}'.format('No.', 'Variable name'))
        for ind, var_name in enumerate(self._variables.keys()):
            mylog.simpleinfo.info('{:^20d}{:30s}'.format(ind+1, var_name))
        mylog.simpleinfo.info('')

    def keys(self) -> list:
        return list(self._variables.keys())

    def items(self) -> dict:
        return dict(self._variables.items())

    def __setitem__(self, key, value):
        if value is not None:
            if not issubclass(value.__class__, self.__variable_model__):
                raise TypeError
        self._variables[key] = value
        self._variables[key].dataset = self

    def __getitem__(self, key) -> __variable_model__:
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
    def exist(self, var_name) -> bool:
        if var_name in self._variables.keys():
            return True
        else:
            return False

    def remove_variable(self, name):
        del self[name]

    def get_variable_names(self) -> list:
        return list(self._variables.keys())

    def register_method(self, func):
        from types import MethodType
        setattr(self, func.__name__, MethodType(func, self))


class DatasetUser(DatasetBase):
    def __init__(self,
                 dt_fr: datetime.datetime = None,
                 dt_to: datetime.datetime = None,
                 name: str = '',
                 visual: str = 'off',
                 label_fields: list = ('name', 'kind'), **kwargs):

        super().__init__(
            dt_fr=dt_fr, dt_to=dt_to, name=name, kind='user-defined', visual=visual, label_fields=label_fields, **kwargs
        )


class DatasetSourced(DatasetBase):

    def __init__(self,
                 dt_fr: datetime.datetime = None,
                 dt_to: datetime.datetime = None,
                 name: str = '',
                 visual: str = 'off',
                 label_fields: list = ('name', 'kind'), **kwargs):

        kind = kwargs.pop('kind', 'sourced')
        super().__init__(
            dt_fr=dt_fr, dt_to=dt_to, name=name, kind=kind, visual=visual, label_fields=label_fields, **kwargs
        )

        self.loader = kwargs.pop('loader', None)
        self.downloader = kwargs.pop('downloader', None)
        self.load_mode = kwargs.pop('load_mode', 'AUTO')  # ['AUTO'], 'dialog', 'assigned'
        self.data_root_dir = kwargs.pop('data_root_dir', pref.datahub_data_root_dir)
        self.data_file_paths = kwargs.pop('data_file_paths', [])
        self.data_file_num = kwargs.pop('data_file_num', 0)
        self.data_file_ext = kwargs.pop('data_file_ext', '*')
        self.data_search_recursive = kwargs.pop('data_search_recursive', False)
        self.time_clip = kwargs.pop('time_clip', True)

    def search_data_files(
            self,
            initial_file_dir=None, search_pattern='*', recursive=None,
            direct_append=True, allow_multiple_files=False, include_extension=True,
            **kwargs) -> list:
        """
        Search the data files by the input pattern in the file name. The search method is based on pathlib.glob.
        For a dataset inheritance, a wrapper can be added for a custom setting.

        :param initial_file_dir: The initial file directory for searching.
        :type initial_file_dir: str or pathlib.Path, default: DatasetModel.data_root_dir.
        :param search_pattern: Unix style pathname pattern, see also pathlib.glob.
        :type search_pattern: str.
        :param recursive: Search recursively if True.
        :type recursive: bool, default: DatasetModel.data_search_recursive.
        :param allow_multiple_files: Allow multiple files as a result.
        :type allow_mulitple_files: bool, default: False.
        :param direct_append: Append directly the searched results to Dataset.data_file_paths.
            If `False`, the file path list is returned only.
        :type direct_append: bool, default: `True`

        :return: a list of the file paths.
        """

        file_paths = []
        if initial_file_dir is None:
            initial_file_dir = self.data_root_dir
        if recursive is None:
            recursive = self.data_search_recursive

        if recursive:
            search_pattern = '**/' + search_pattern

        if include_extension and (self.data_file_ext not in ['*', '.*']):
            if isinstance(self.data_file_ext, str):
                exts = [self.data_file_ext]
            elif isinstance(self.data_file_ext, (list, tuple)):
                exts = self.data_file_ext
            else:
                raise TypeError
            for i, ext in enumerate(exts):
                if ext[0] != '.':
                    exts[i] = '.' + ext
            paths = [p.resolve() for p in initial_file_dir.glob(search_pattern) if p.suffix in exts]
            #
            # if str(self.data_file_ext):
            #     search_pattern = search_pattern + '.' + self.data_file_ext
        else:
            paths = list(initial_file_dir.glob(search_pattern))

        import natsort
        paths = natsort.natsorted(paths, reverse=False)
        if len(paths) == 1:
            file_paths.extend(paths)
        elif len(paths) > 1:
            if allow_multiple_files:
                file_paths.extend(paths)
            else:
                mylog.StreamLogger.error("Multiple files found! Restrict the search condition.")
                print(paths)
                raise FileExistsError
        else:
            mylog.simpleinfo.info('Cannot find the requested data file in {}'.format(initial_file_dir))
        if direct_append:
            self.data_file_paths.extend(file_paths)
        return file_paths

    def open_dialog(self, initial_file_dir: str = None, data_file_num: int = None, **kwargs):
        """
        Open a dialog to select the data files.
        """
        import tkinter as tk
        from tkinter import filedialog
        from tkinter import simpledialog
        def tk_open_file():

            root = tk.Tk()
            root.withdraw()

            dialog_title = "Select a data file ..."

            if str(self.data_file_ext):
                p1 = '*.' + self.data_file_ext
            file_types = (('data files', p1), ('all files', '*.*'))
            file_name = filedialog.askopenfilename(
                title=dialog_title,
                initialdir=initial_file_dir,
                filetypes=file_types,
            )

            return file_name

        root = tk.Tk()
        root.withdraw()
        self.data_file_num = kwargs.pop('data_file_num', self.data_file_num)
        if self.data_file_num == 0:
            self.data_file_num = simpledialog.askinteger('Input dialog', 'Input the number of files:', initialvalue=1)

        for nf in range(self.data_file_num):
            title = f"Select File {nf+1}: "
            file_types = (('eiscat files', '*.' + self.data_file_ext), ('all files', '*.*'))
            file_name = filedialog.askopenfilename(
                title=title,
                initialdir=initial_file_dir
            )
            self.data_file_paths.append(pathlib.Path(file_name))

    def check_data_files(self, load_mode: str = None, **kwargs):
        """
        Check the existing of the data files before loading the data, depending on the loading mode (``load_mode``).
        This methods still needs to be improved as different datasets may have different variables as epochs. Two kinds
        of things can be done: 1. write a wrapper in the new dataset inheritance. 2. Add a script to recognize the
        epoch variables.

        """
        if load_mode is not None:
            self.load_mode = load_mode
        if list(self.data_file_paths):
            self.load_mode = 'assigned'
        if self.load_mode == 'AUTO':
            self.search_data_files(**kwargs)
        elif self.load_mode == 'dialog':
            self.open_dialog(**kwargs)
        elif self.load_mode == 'assigned':
            self.data_file_paths = kwargs.pop('data_file_paths', self.data_file_paths)
            self.data_file_paths = [pathlib.Path(f) for f in self.data_file_paths]
        else:
            raise NotImplementedError
        self.data_file_num = len(self.data_file_paths)

    def time_filter_by_range(self, var_datetime=None, var_datetime_name=None):
        """
        Clip the times.
        :param var_datetime:
        :param var_datetime_name:
        :return:
        """
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
            mylog.StreamLogger.warning("Data within the requested time range are not available!")
            return
        if var_datetime is None:
            var_datetime = self['DATETIME']

        shape_0 = var_datetime.value.shape[0]
        for var in self._variables.values():
            if var.value is None:
                continue
            if var.value.shape[0] == shape_0 and len(var.value.shape) > 1:
                var.value = var.value[inds, ::]

    def get_time_ind(self, ut, time_res=None, var_datetime=None, var_datetime_name=None, edge_cutoff=True):
        if var_datetime is None and var_datetime_name is None:
            var_datetime = self['DATETIME']
        if var_datetime_name is not None:
            var_datetime = self[var_datetime_name]
        if var_datetime.value is None:
            return
        dts = var_datetime.value.flatten()

        if time_res is not None:
            edge_cutoff = False

        ind = []
        if edge_cutoff:
            if ut > dts[-1] or ut < dts[0]:
                mylog.StreamLogger.warning('The input time is out of the range! Set "edge_cutoff=False" if needed!')
                return ind
        delta_sectime = [delta_t.total_seconds() for delta_t in (dts - ut)]
        
        ind = np.where(np.abs(delta_sectime) == np.min(np.abs(delta_sectime)))[0][0]

        if np.abs((dts[ind] - ut).total_seconds()) > time_res:
            mylog.StreamLogger.warning('The input time does not match any time in the list! Check the time resolution ("time_res") in seconds!')
            return []
        return ind

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


class LoaderBase(object):
    """
    Loader Base for loading data from the assigned files
    """
    def __init__(self):
        self.variables = {}
        self.metadata = {}

    def load_data(self):
        raise NotImplementedError


class DownloaderBase(object):
    """
    Downloader Base for downloading data files from different data sources

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
