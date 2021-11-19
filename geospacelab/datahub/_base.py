# Licensed under the BSD 3-Clause License
# Copyright (C) 2021 GeospaceLab (geospacelab)
# Author: Lei Cai, Space Physics and Astronomy, University of Oulu

__author__ = "Lei Cai"
__copyright__ = "Copyright 2021, GeospaceLab"
__license__ = "BSD-3-Clause License"
__email__ = "lei.cai@oulu.fi"
__docformat__ = "reStructureText"

import pathlib

# from geospacelab.datahub.variable_base import *
from geospacelab import preferences as pref

import geospacelab.toolbox.utilities.pyclass as pyclass
import geospacelab.toolbox.utilities.pybasic as pybasic
import geospacelab.toolbox.utilities.pylogging as mylog


class DatasetModel(object):
    """
    A dataset is a dictionary-like object used for downloading and loading data from a data source. The items in
    the dataset are the variables loaded from the data files.

    :param name: The name of the dataset.
    :type name: str or None.
    :param kind: The type of the dataset. 'sourced': the data source has been added in the package,
    'temporary': a dataset added temporarily, 'user-defined': a data source defined by the user.
    :type kind: {'sourced', 'temporary', 'user-defined'}
    :param dt_fr: the starting time of the data records.
    :type dt_fr: datetime.datetime, default: None.
    :param dt_to: the stopping time of the the data records.
    :param load_mode: The mode for the dataset to load the data. "AUTO": Automatically searching the data files and
    load the data, ''
    :type dt_to: datetime.datetime, default: None.
    :param loader: the loader class used to load the data.
    :type loader: LoaderModel.
    :param downloader: the downloader class used to download the data.
    :type downloader: DownloaderModel

    """
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


class VariableModel(object):
    """VariableModel is a base class for a geospace variable with useful attributes

    :param  name: The variable's name, ['']
    :type   name: str
    :param  fullname: The variable's full name, ['']
    :type   fullname: str
    :param  label: The variable's label for display. If a raw string (e.g., r'$\alpha$'),
        it will show a latex format.
    :type   label: str
    :param  data_type: The variable's data type in one of them: 'support_data', 'data', and 'metadata'
        in the same format as in a NASA's cdf file. ['']
    :type   data_type: str
    :param  group: The group that the variable is belonged to, e.g., var.name = 'v_i_z', var.group = 'ion velocity',
        as the label in plots with multiple lines. ['']
    :type   group: str
    :param  unit: The variable's unit. ['']
    :type   unit: str
    :param  unit_label: The unit's  label,  used for plots. The string is a raw string (e.g., r'$n_e$').
                If None, the plot will use unit as a label.
    :type   unit_label: str
    :param  quantity: The physical quantity associated with the variable, waiting for implementing. [None]
    :type   quantity: TBD.
    :param  value: the variable's value. Usually it's a np.ndarray. The axis=0 along the time, axis=1 along height, lat,
    lon. For a scalar, value in a shape of (1, ). [None]
    :type   value: np.ndarray
    :param  error: the variable's error. Either a np.ndarray or a string. When it's a string, the string is a variable name
    indicating the variable in the associated dataset (see :attr:`dataset` below).
    :type
    :param
    :type
    :param
    :type
    :param
    :type
    :param
    :type
    :param
    :type
    :param
    :type
    :param
    :type
    """

    def __init__(self, **kwargs):
        """Initial settings

        :params:
        """
        # set default values

        self.name = kwargs.pop('name', '')
        self.fullname = kwargs.pop('fullname', '')

        self.label = kwargs.pop('label', '')

        self.data_type = kwargs.pop('data_type', None)  # 'support_data', 'data', 'metadata'
        self.group = kwargs.pop('group', '')

        self.unit = kwargs.pop('unit', None)
        self.unit_label = kwargs.pop('unit_label', None)

        self.quantity = kwargs.pop('quantity', None)

        self.value = kwargs.pop('value', None)
        self.error = kwargs.pop('error', None)

        self.variable_type = kwargs.pop('variable_type', 'scalar')  # scalar, vector, tensor, ...
        self.ndim = kwargs.pop('ndim', None)
        self._depends = {}
        self.depends = kwargs.pop('depends', None)

        self.dataset = kwargs.pop('dataset', None)

        self._visual = None
        self.visual = kwargs.pop('visual', 'off')
        self._attrs_registered = ['name', 'fullname', 'label', 'data_type', 'group', 'unit', 'unit_label',
                                  'quantity', 'value', 'error', 'variable_type', 'ndim', 'depends', 'dataset',
                                  'visual']

    def config(self, logging=True, **kwargs):
        pyclass.set_object_attributes(self, append=False, logging=logging, **kwargs)

    def add_attr(self, logging=True, **kwargs):
        self._attrs_registered.extend(kwargs.keys())
        pyclass.set_object_attributes(self, append=True, logging=logging, **kwargs)

    def clone(self, omit_attrs=None):
        if omit_attrs is None:
            omit_attrs = {}
        kwargs = {}
        for key in self._attrs_registered:
            if key in omit_attrs.keys():
                continue
            if key == 'visual':
                kwargs['visual'] = self.visual.clone()
            elif key == 'dataset':
                kwargs['dataset'] = self.dataset
            else:
                kwargs[key] = copy.deepcopy(getattr(self, key))
        return self.__class__(**kwargs)

    def get_depend(self, axis=None, retrieve_data=True):
        # axis = 0, 1, 2
        if axis in self.depends.keys():
            depend = self.depends[axis]
        else:
            return None
        depend_new = copy.deepcopy(depend)
        if retrieve_data:
            for key, value in depend.items():
                if isinstance(value, str):
                    try:
                        value = self.dataset[value].value
                    except KeyError:
                        print('The variable {} has not been assigned!'.format(key))
                        value = None
                depend_new[key] = copy.deepcopy(value)
        return depend_new

    def set_depend(self, axis, depend_dict):
        if not isinstance(depend_dict, dict):
            raise TypeError
        self.depends[axis] = depend_dict

    def get_attr_from_string(self, string):
        if not str(string):
            return string
        if string[0] == '@':
            splits = string[1:].split('.')
            if splits[0] in ['v']:
                result = getattr(self, splits[1])
            elif splits[0] in ['d']:
                result = getattr(self.dataset[splits[1]], splits[2])
            else:
                raise ValueError
        else:
            result = string
        return result

    def get_visual_axis_attr(self, attr_name=None, axis=None):
        attr = getattr(self.visual.axis[axis], attr_name)
        type_attr = type(attr)
        if type_attr is not list:
            attr = [attr]
        results = []
        for a in attr:
            if type(a) is str:
                result = self.get_attr_from_string(a)
            else:
                result = a
            results.append(copy.deepcopy(result))
        if type_attr is not list:
            results = results[0]
        return results

    def join(self, var_new):
        if issubclass(var_new.__class__, VariableModel):
            v = var_new.value
        else:
            v = var_new

        if self.value is None:
            self.value = v
            return

        if type(v) is np.ndarray:
            self.value = np.concatenate((self.value, v), axis=0)
        else:
            if np.isscalar(v):
                v = tuple([v])
            elif isinstance(v, (list, tuple)):
                v = tuple(v)
            if v != self.value:
                mylog.StreamLogger.warning("The scalar variables have different values!")
            return

    def __repr__(self):
        value_repr = repr(self.value)
        rep = f"GeospaceLab Variable object <name: {self.name}, value: {value_repr}, unit: {self.unit}>"
        return rep

    @property
    def visual(self):
        return self._visual

    @visual.setter
    def visual(self, value):
        if value == 'new':
            self._visual = Visual(variable=self)
        elif value == 'on':
            if not isinstance(self._visual, Visual):
                self.visual = 'new'
        elif isinstance(value, Visual):
            self._visual = value
        elif type(value) is dict:
            if not isinstance(self._visual, Visual):
                self.visual = 'new'
            pyclass.set_object_attributes(self.visual, append=False, logging=True, **value)
        elif value == 'off':
            self._visual = None
        else:
            raise ValueError

    @property
    def dataset(self):
        if self._dataset_ref is None:
            return None
        else:
            return self._dataset_ref()

    @dataset.setter
    def dataset(self, dataset_obj):
        if dataset_obj is None:
            self._dataset_ref = None
            return

        from geospacelab.datahub.dataset_base import DatasetModel
        if issubclass(dataset_obj.__class__, DatasetModel):
            self._dataset_ref = weakref.ref(dataset_obj)
        else:
            raise TypeError

    @property
    def value(self):
        if self._value is None:
            return None
        elif isinstance(self._value, str):
            if self.dataset is not None:
                return self.dataset[self._value].value
            else:
                return self._value
        else:
            return self._value

    @value.setter
    def value(self, v):
        if v is None:
            self._value = None
            return
        # type check
        if type(v) is str:
            self._value = v
            return
        if not isinstance(v, np.ndarray):
            if basic.isnumeric(v):
                v = tuple([v])
            elif isinstance(v, (list, tuple)):
                v = tuple(v)
            else:
                raise TypeError
            self._value = v
            return
            # reshape np.array with shape like (m,) m>1
        # if len(v.shape) == 1 and v.shape != (1,):
        #     v = v.reshape((v.shape[0], 1))
        self._value = v

    @property
    def error(self):
        if self._error is None:
            return None
        elif isinstance(self._error, str):
            if self.dataset is not None:
                return self.dataset[self._error].value
            else:
                return self._error
        else:
            return self._error

    @error.setter
    def error(self, v):
        if v is None:
            self._error = None
            return
        # type check
        if type(v) is str:
            self._error = v
            return
        if not isinstance(v, np.ndarray):
            if basic.isnumeric(v):
                v = np.array([v])
            elif isinstance(v, (list, tuple)):
                v = np.array(v)
            else:
                raise TypeError
        # reshape np.array with shape like (m,) m>1
        if len(v.shape) == 1 and v.shape != (1,):
            v = v.reshape((v.shape[0], 1))
        self._error = v

    @property
    def unit_label(self):
        label = self._unit_label
        if not str(label):
            label = self.unit
        return label

    @unit_label.setter
    def unit_label(self, label):
        if label is None:
            self._unit_label = ''
            return
        if type(label) is not str:
            raise TypeError
        self._unit_label = label

    @property
    def label(self):
        lb = self._label
        if not str(lb):
            lb = self.name
        return lb

    @label.setter
    def label(self, lb):
        if lb is None:
            self._label = ''
            return
        if type(lb) is not str:
            raise TypeError
        self._label = lb

    @property
    def variable_type(self):
        return self._variable_type

    @variable_type.setter
    def variable_type(self, value):
        if value is None:
            self._variable_type = None
            return
        if value.lower() not in ['scalar', 'vector', 'matrix', 'tensor']:
            raise AttributeError
        self._variable_type = value.lower()

    @property
    def ndim(self):
        if self._ndim is None:
            var_type = self.variable_type
            if var_type is None:
                mylog.StreamLogger.warning('the variable type has not been defined! Use default type "scalar"...')
                var_type = "scalar"
            value = self.value
            if value is None:
                mylog.StreamLogger.warning("Variable's value was not assigned. Return None!")
                return None
            shape = value.shape
            offset = 0
            if var_type == 'scalar':
                offset = -1
            elif var_type == 'vector':
                offset = -1
            elif var_type == 'matrix':
                offset = -2
            if shape == (1,):
                nd = 0
            else:
                nd = len(shape) + offset
            if nd <= 0:
                raise ValueError("ndim cannot be a non-positive integer")
            self._ndim = nd
        return self._ndim

    @ndim.setter
    def ndim(self, value):
        if value is None:
            self._ndim = None
            return
        if type(value) is not int:
            raise TypeError
        self._ndim = value

    @property
    def depends(self):
        return self._depends

    @depends.setter
    def depends(self, d_dict):
        if d_dict is None:
            return

        if type(d_dict) is not dict:
            raise TypeError

        for key, value in d_dict.items():
            if type(key) is not int:
                raise KeyError
            if value is None:
                self._depends[key] = {}
            else:
                self._depends[key] = value


class VisualAxis(object):
    def __init__(self):
        self.data = None
        self.data_err = None
        self.data_scale = 1.
        self.data_res = None
        self.mask_gap = None
        self.label = ''
        self.label_style = 'double'  # or 'single
        self.label_pos = None  # label position
        self.unit = ''
        self.lim = None
        self.scale = 'linear'
        self.invert = False
        self.ticks = None
        self.tick_labels = None
        self.minor_ticks = None
        self.major_tick_max = 6
        self.minor_tick_max = None
        self.visible = True

    def config(self, logging=True, **kwargs):
        pyclass.set_object_attributes(self, append=False, logging=logging, **kwargs)


class VisualPlotConfig(object):
    def __init__(self):
        self.visible = True
        self.line = {}
        self.errorbar = {}
        self.pcolormesh = {}
        self.imshow = {}
        self.scatter = {}
        self.legend = {}
        self.style = None

    def config(self, logging=True, **kwargs):
        pyclass.set_object_attributes(self, append=False, logging=logging, **kwargs)


class Visual(object):

    def __init__(self, **kwargs):
        self._axis = {}
        self.variable = None
        self._ndim = None

        self.variable = kwargs.pop('variable', None)
        self.axis = kwargs.pop('axis', None)
        self.plot_config = kwargs.pop('plot_config', VisualPlotConfig())
        self.ndim = kwargs.pop('ndim', self._ndim)

    def config(self, logging=True, **kwargs):
        pyclass.set_object_attributes(self, append=False, logging=logging, **kwargs)

    def add_attr(self, logging=True, **kwargs):
        pyclass.set_object_attributes(self, append=True, logging=logging, **kwargs)

    def clone(self):
        kwargs = {
            'variable': self.variable,
            'axis': copy.deepcopy(self.axis),
            'plot_config': copy.deepcopy(self.plot_config),
            'ndim': copy.deepcopy(self.ndim)
        }

        return self.__class__(**kwargs)

    @property
    def variable(self):
        if self._variable_ref is None:
            return None
        else:
            return self._variable_ref()

    @variable.setter
    def variable(self, var_obj):
        if var_obj is None:
            self._variable_ref = None
            return
        if issubclass(var_obj.__class__, VariableModel):
            self._variable_ref = weakref.ref(var_obj)
        else:
            raise TypeError

    @property
    def ndim(self):
        _ndim = self._ndim
        if _ndim is None:
            if self.variable is not None:
                _ndim = self.variable.ndim
        return _ndim

    @ndim.setter
    def ndim(self, value):
        self._ndim = value

    @property
    def axis(self):
        if not dict(self._axis):
            ndim = self.ndim
            if ndim is None:
                try:
                    ndim = self.variable.ndim
                    if ndim is None:
                        return None
                    self.ndim = ndim
                except AttributeError:
                    return None
            if ndim < 2:
                ndim = 2
            for ind in range(ndim + 1):
                self._axis[ind] = VisualAxis()
        return self._axis

    @axis.setter
    def axis(self, a_dict):
        if a_dict is None:
            return
        if type(a_dict) is dict:
            for key, value in a_dict.items():
                if value is None:
                    self._axis[key] = VisualAxis()
                elif type(value) is dict:
                    self._axis[key].config(**value)
                elif isinstance(value, VisualAxis):
                    self._axis[key] = value
                else:
                    raise TypeError
        else:
            raise TypeError

    @property
    def plot_config(self):
        return self._plot

    @plot_config.setter
    def plot_config(self, value):
        if value is None:
            self._plot = VisualPlotConfig()
            return
        if isinstance(value, VisualPlotConfig):
            self._plot = value
        elif type(value) is dict:
            self._plot.config(**value)
        else:
            raise TypeError


class Visual_1(object):
    def __init__(self, **kwargs):
        self.plot_type = None
        self.x_data = None
        self.y_data = None
        self.z_data = None
        self.x_data_scale = 1
        self.y_data_scale = 1
        self.z_data_scale = 1
        self.x_data_res = None
        self.y_data_res = None
        self.z_data_res = None
        self.x_err_data = None
        self.y_err_data = None
        self.z_err_data = None
        self.x_lim = None
        self.y_lim = None
        self.z_lim = None
        self.x_label = None
        self.y_label = None
        self.z_label = None
        self.x_scale = None
        self.y_scale = None
        self.z_scale = None
        self.x_unit = None
        self.y_unit = None
        self.z_unit = None
        self.x_ticks = None
        self.y_ticks = None
        self.z_ticks = None
        self.x_tick_labels = None
        self.y_tick_labels = None
        self.z_tick_labels = None
        self.color = None
        self.visible = True
        self.plot_config = {}
        self.axis_config = {}
        self.legend_config = {}
        self.colorbar_config = {}
        self.errorbar_config = {}
        self.pcolormesh_config = {}
        self.config(**kwargs)

    def config(self, logging=True, **kwargs):
        pyclass.set_object_attributes(self, append=False, logging=logging, **kwargs)

    def add_attr(self, logging=True, **kwargs):
        pyclass.set_object_attributes(self, append=True, logging=logging, **kwargs)

