# Licensed under the BSD 3-Clause License
# Copyright (C) 2021 GeospaceLab (geospacelab)
# Author: Lei Cai, Space Physics and Astronomy, University of Oulu

__author__ = "Lei Cai"
__copyright__ = "Copyright 2021, GeospaceLab"
__license__ = "BSD-3-Clause License"
__email__ = "lei.cai@oulu.fi"
__docformat__ = "reStructureText"

import copy
import weakref
import re

import numpy as np
from typing import Dict

import geospacelab.toolbox.utilities.pyclass as pyclass
import geospacelab.toolbox.utilities.pylogging as mylog
import geospacelab.toolbox.utilities.pybasic as basic


class VisualAxis(object):
    """
    The attribute class appended to the Visual class for setting the attributes along an axis.


    """

    def __init__(self):
        self.data = None # "@v.value"
        self.data_err = None 
        self.data_scale = 1.
        self.data_res = None
        self.mask_gap = None
        self.label = None # "@v.label"
        self.label_style = 'double'  # or 'single
        self.label_pos = None  # label position
        self.unit = '' 
        self.lim = None
        self.scale = 'linear'
        self.invert = False
        self.ticks = None
        self.tick_labels = None
        self.minor_ticks = None
        self.major_tick_min = None
        self.major_tick_max = None
        self.minor_tick_max = None
        self.minor_tick_min = None
        self.reverse = False
        self.visible = True
        self.shift = None

    def config(self, logging=True, **kwargs):
        pyclass.set_object_attributes(self, append=False, logging=logging, **kwargs)


class VisualPlotConfig(object):
    """
    The attribute class appended to the Visual class for setting the plots.


    """

    def __init__(self):
        self.visible = True
        self.line = {}
        self.pattern = {}
        self.errorbar = {}
        self.pcolormesh = {}
        self.imshow = {}
        self.scatter = {}
        self.legend = {}
        self.colorbar = {}
        self.bar = {}
        self.scatter = {}
        self.fill_between = {}
        self.style = None # "1E", "1P" or "1noE", "1S", "1B", "2P"

    def config(self, logging=True, **kwargs):
        pyclass.set_object_attributes(self, append=False, logging=logging, **kwargs)


class NDim(int):
    def __new__(cls, value, extra):
        return float.__new__(cls, value)

    def __init__(self, value, extra):
        float.__init__(value)
        self.extra = extra


class Visual(object):
    """
    The Visual class is used for setting the visualization attributes appending to a variable.


    """

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
        if issubclass(var_obj.__class__, VariableBase):
            self._variable_ref = weakref.ref(var_obj)
        else:
            raise TypeError

    @property
    def ndim(self) -> int:
        _ndim = self._ndim
        if _ndim is None:
            if self.variable is not None:
                _ndim = self.variable.ndim
        return _ndim

    @ndim.setter
    def ndim(self, value):
        self._ndim = value

    @property
    def axis(self) -> Dict[int, VisualAxis]:
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
                if not type(key) is int:
                    raise TypeError
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


class VariableBase(object):
    """VariableModel is a base class for a geospace variable with useful attributes

    :ivar str name: The variable's name.
    :ivar str fullname: The variable's full name.
    :ivar str  label: The variable's label for display. If a raw string (e.g., r'$\alpha$'),
        it will show a latex format.
    :ivar str data_type: The variable's data type in one of them: 'support_data', 'data', and 'metadata'
        in the same format as in a NASA's cdf file.
    :ivar str group: The group that the variable is belonged to, e.g., var.name = 'v_i_z', var.group = 'ion velocity',
        as the label in plots with multiple lines. ['']
    :ivar str  unit: The variable's unit. ['']
    :ivar str  unit_label: The unit's  label,  used for plots. The string is a raw string (e.g., r'$n_e$').
                If None, the plot will use unit as a label.
    :ivar Quantity object  quantity: The physical quantity associated with the variable, waiting for implementing. [None]
    :ivar np.ndarray  value: the variable's value.  The default type is a np.ndarray.
        The axis=0 along the time, axis=1 along height, lat,
        lon. For a scalar, value in a shape of (1, ).
    :ivar str or np.ndarray  error: the variable's error. Either a np.ndarray or a string. When it's a string, the string is a variable name
        indicating the variable in the associated dataset (see :attr:`dataset` below).
    :ivar int ndim: The number of dimension.
    :ivar dict depends: The full depends of the variables. Usually Axis 0 for time, next for spatial distributions,
        and then for components.
    :ivar Dataset object dataset: The dataset that the variable is appended.
    :ivar Visual object visual: the attributes for visualization.

    """

    __visual_model__ = Visual
    __dataset_model__ = None

    def __init__(
            self, value=None, error=None, data_type=None,
            name='', fullname='', label='', group='',
            unit='', unit_label=None, quantity=None,
            variable_type='scalar',
            ndim=None, depends=None, dataset=None, visual=None,
            **kwargs):
        """Initial settings

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
                lon. For a scalar, value in a shape of (1, ).
            :type   value: np.ndarray, default: None
            :param  error: the variable's error. Either a np.ndarray or a string. When it's a string, the string is a variable name
                indicating the variable in the associated dataset (see :attr:`dataset` below).
            :type error: str or np.ndarry
            :param ndim: The number of dimension.
            :type ndim: int
            :param depends: The full depends of the variables. Usually Axis 0 for time, next for spatial distributions,
                and then for components.
            :type depends: dict
            :param dataset: The dataset that the variable is appended.
            :type dataset: DatasetModel object
            :param visual: the attributes for visualization.
            :type visual: dict or Visual object, default: None.
        """
        # set default values

        from geospacelab.datahub.__dataset_base__ import DatasetBase

        self.__dataset_model__ = DatasetBase

        self.name = name
        self.fullname = fullname

        self.label = label

        self.data_type = data_type  # 'support_data', 'data', 'metadata'
        self.group = group

        self.unit = unit
        self.unit_label = unit_label

        self.quantity = quantity

        self.value = value
        self.error = error

        self.variable_type = variable_type  # scalar, vi, tensor, ...
        self.ndim = ndim
        self._depends = {}
        self.depends = depends

        self.dataset = dataset

        self._visual = None
        self.visual = visual
        self._attrs_registered = ['name', 'fullname', 'label', 'data_type', 'group', 'unit', 'unit_label',
                                  'quantity', 'value', 'error', 'variable_type', 'ndim', 'depends', 'dataset',
                                  'visual']

    def config(self, logging=True, **kwargs):
        """
        Configure the variable attributes. If the attribute is not the default attributes, return error.

        :param logging: If True, show logging.
        :param kwargs: A dictionary of the attributes.
        :return: None
        """
        pyclass.set_object_attributes(self, append=False, logging=logging, **kwargs)

    def add_attr(self, logging=True, **kwargs):
        """
        Similar as config, but add a new attribute.
        """

        self._attrs_registered.extend(kwargs.keys())
        pyclass.set_object_attributes(self, append=True, logging=logging, **kwargs)

    def clone(self, omit_attrs=None, var_name=None):
        """
        Clone a variable and return a new instance.

        :param omit_attrs: The attributes omitted to be copied. If None, copy all.
        :return: new variable instance
        """
        if omit_attrs is None:
            omit_attrs = []
        kwargs = {}
        for key in self._attrs_registered:
            if key in omit_attrs:
                continue
            if key == 'visual':
                if self.visual is not None:
                    kwargs['visual'] = self.visual.clone()
            elif key == 'dataset':
                kwargs['dataset'] = self.dataset
            else:
                kwargs[key] = copy.deepcopy(getattr(self, key))
        var_new = self.__class__(**kwargs)
        if var_name is not None:
            self.dataset[var_name] = var_new
        return var_new

    def get_depend(self, axis=None, retrieve_data=True):
        """
        Get the dependence of the variable along an axis.

        :param axis: The axis.
        :param retrieve_data: If True, Convert identifier to data.
        :return: A dictionary of dependence.
        """
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
        """
        Set the dependence along an axis.

        :param axis: The axis.
        :param depend_dict: the dictionary of the dependence, e.g., {'DATETIME': 'DATETIME'}.
        :return:
        """
        if not isinstance(depend_dict, dict):
            raise TypeError
        self.depends[axis] = depend_dict

    def _string_parse_for_attr(self, string):
        if not str(string):
            return string
        rc = re.compile(r'^@(.+)')
        rm = rc.match(string)
        if rm is None:
            return string

        results = rm.groups()[0].split('.')

        if results[0].lower() in ['v', 'var', 'variable']:
            attr = getattr(self, results[1])
        elif results[0].lower() in ['d', 'dataset']:
            if len(results) == 2:
                results.append('value')
            attr = getattr(self.dataset[results[1]], results[2])
        elif results[0].lower() in ['vd', 'depends']:
            depend = self.get_depend(axis=int(results[1]))
            attr = depend[results[2]]
        return attr
        #
        # if string[0] == '@':
        #     splits = string[1:].split('.')
        #     if splits[0] in ['v']:
        #         result = getattr(self, splits[1])
        #     elif splits[0] in ['d']:
        #         result = getattr(self.dataset[splits[1]], splits[2])
        #     else:
        #         raise ValueError
        # else:
        #     result = string
        return result

    def get_visual_axis_attr(self, attr_name=None, axis=None):
        """
        Get the visual attributes along an axis.

        :param attr_name:
        :param axis:
        :return:
        """
        attr = getattr(self.visual.axis[axis], attr_name)
        type_attr = type(attr)
        if type_attr is not list:
            attr = [attr]
        results = []
        for a in attr:
            if type(a) is str:
                result = self._string_parse_for_attr(a)
            else:
                result = a
            results.append(copy.deepcopy(result))
        if type_attr is not list:
            results = results[0]
        return results

    def join(self, var_new):
        """
        Join a numpy array or a variable instance to the current variable.
        :param var_new: The input variable.
        :type var_new: list, np.ndarray, or variable instance
        :return:
        """
        if issubclass(var_new.__class__, VariableBase):
            v = copy.deepcopy(var_new.value)
        else:
            v = copy.deepcopy(var_new)

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

    def flatten(self):
        return self.value.flatten()

    def __repr__(self):
        value_repr = repr(self.value)
        rep = f"GeospaceLab Variable object <name: {self.name}, value: {value_repr}, unit: {self.unit}>"
        return rep

    @property
    def visual(self) -> __visual_model__:
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
        elif value == 'off' or value is None:
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

        if issubclass(dataset_obj.__class__, self.__dataset_model__):
            self._dataset_ref = weakref.ref(dataset_obj)
        else:
            raise TypeError

    @property
    def value(self) -> np.ndarray:
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
    def error(self) -> np.ndarray:
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
        if label is None:
            return self.unit
        else:
            return label

    @unit_label.setter
    def unit_label(self, label: str or None):
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
        if value.lower() not in ['scalar', 'vi', 'matrix', 'tensor']:
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
            elif var_type == 'vi':
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
            self._depends = {}
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


#
#
# class VariableBase(np.ndarray):
#     _attrs_registered = [
#         'name', 'fullname', 'label',
#         'data_type', 'group',
#         'unit', 'unit_label',
#         'quantity',
#         'value', 'error',
#         'variable_type',
#         'kind',
#         'depends', 'dataset',
#         'visual'
#     ]
#
#     def __new__(cls, arr, copy=True, dtype=None, order='C', subok=False, ndmin=0, **kwargs):
#         if issubclass(arr.__class__, VariableBase):
#             obj_out = arr.clone()
#         else:
#             obj_out = np.array(arr, copy=copy, dtype=dtype, order=order, subok=subok, ndmin=ndmin)
#             obj_out = obj_out.view(cls)
#
#         obj_out.config(**kwargs)
#
#         return obj_out
#
#     def config(self, logging=True, **kwargs):
#         """
#         Configure the variable attributes. If the attribute is not the default attributes, return error.
#
#         :param logging: If True, show logging.
#         :param kwargs: A dictionary of the attributes.
#         :return: None
#         """
#
#         # check registered attributes
#         for key, value in kwargs.items():
#             if key not in self._attrs_registered:
#                 raise KeyError(f'The attribute {key} is not registered! Use "add_attr()" to add a new attribute.')
#
#         pyclass.set_object_attributes(self, append=False, logging=logging, **kwargs)
#
#     def add_attr(self, logging=True, **kwargs):
#         """
#         Similar as config, but add a new attribute.
#         """
#
#         self._attrs_registered.extend(kwargs.keys())
#         pyclass.set_object_attributes(self, append=True, logging=logging, **kwargs)
#
#     def copy_attr(self, obj, to_whom=None, omit_attrs=None):
#         """
#         Copy the attributes from obj to the current instance.
#
#         :param obj: the VariableBase or its subclass instance
#         :type obj: VariableBase instance
#         :param to_whom: The destination instance
#         :type to_whom: VariableBase instance. If ``None``, copy to self
#         :param omit_attrs: The attributes omitted. If None, copy all
#         :type omit_attrs: dict or None
#         """
#
#         if omit_attrs is None:
#             omit_attrs = {}
#
#         if to_whom is None:
#             to_whom = self
#
#         for key in self._attrs_registered:
#             if key in omit_attrs.keys():
#                 continue
#             if key == 'visual':
#                 to_whom.visual = obj.visual.clone()
#             elif key == 'dataset':
#                 to_whom.dataset = obj.dataset
#             else:
#                 setattr(to_whom, key, copy.deepcopy(getattr(obj, key)))
#
#         return to_whom
#
#     def clone(self, omit_attrs=None):
#         """
#         Clone a variable and return a new instance.
#
#         :param omit_attrs: The attributes omitted to be copied. If None, copy all.
#         :return: new variable instance
#         """
#
#         obj_new = self.__class__(self.view(np.ndarray))
#         obj_new.copy_attr(self, omit_attrs=omit_attrs)
#
#         return obj_new
#
#     def __array_ufunc__(self, ufunc, method, *inputs, out=None, **kwargs):
#         """
#         This implementation of __array_ufunc__ makes sure that all custom attributes are maintained when a ufunc operation is performed on our class.'''
#         See also :ref:`Propagate attributes <https://stackoverflow.com/questions/51520630/subclassing-numpy-array-propagate-attributes>`.
#         """
#
#         # convert inputs and outputs of class ArraySubclass to np.ndarray to prevent infinite recursion
#         args = ((i.view(np.ndarray) if issubclass(i, VariableBase) else i) for i in inputs)
#         outputs = kwargs.pop('out', None)
#         if outputs:
#             kwargs['out'] = tuple((o.view(np.ndarray) if issubclass(o, VariableBase) else o) for o in outputs)
#         else:
#             outputs = (None,) * ufunc.nout
#
#         # call numpys implementation of __array_ufunc__
#         results = super().__array_ufunc__(ufunc, method, *args, **kwargs)  # pylint: disable=no-member
#         if results is NotImplemented:
#             return NotImplemented
#         if method == 'at':
#             # method == 'at' means that the operation is performed in-place. Therefore, we are done.
#             return
#         # now we need to make sure that outputs that where specified with the 'out' argument are handled corectly:
#         if ufunc.nout == 1:
#             results = (results,)
#         results = tuple((self._copy_attrs_to(result) if output is None else output)
#                         for result, output in zip(results, outputs))
#         return results[0] if len(results) == 1 else results
#
#     def __array_finalize__(self, obj):
#         """
#         Used for handling new instances created by three ways:
#         * Explicit constructor:
#             * self type is ´´cls´´
#             * obj type is ``None``
#         * View casting:
#             * self type is ``cls``
#             * obj type is ``np.ndarray``
#         * Slicing (new from template)
#             * self type is ``cls``
#             * obj type is ``cls``
#         """
#
#         # Nothing needed for new variable or np.ndarray.view()
#         if obj is None or isinstance(obj.__class__, np.ndarray):
#             return None
#
#         if issubclass(obj.__class__, self.__class__):
#             self.copy_attr(obj)
#         # else:
#         #     for attr in getattr(self.__class__, '_attrs_registered'):
#         #         self.__setattr__(attr, cp.deepcopy(getattr(self.__class__, attr))
#
#
#
#
# class Visual_1(object):
#     def __init__(self, **kwargs):
#         self.plot_type = None
#         self.x_data = None
#         self.y_data = None
#         self.z_data = None
#         self.x_data_scale = 1
#         self.y_data_scale = 1
#         self.z_data_scale = 1
#         self.x_data_res = None
#         self.y_data_res = None
#         self.z_data_res = None
#         self.x_err_data = None
#         self.y_err_data = None
#         self.z_err_data = None
#         self.x_lim = None
#         self.y_lim = None
#         self.z_lim = None
#         self.x_label = None
#         self.y_label = None
#         self.z_label = None
#         self.x_scale = None
#         self.y_scale = None
#         self.z_scale = None
#         self.x_unit = None
#         self.y_unit = None
#         self.z_unit = None
#         self.x_ticks = None
#         self.y_ticks = None
#         self.z_ticks = None
#         self.x_tick_labels = None
#         self.y_tick_labels = None
#         self.z_tick_labels = None
#         self.color = None
#         self.visible = True
#         self.plot_config = {}
#         self.axis_config = {}
#         self.legend_config = {}
#         self.colorbar_config = {}
#         self.errorbar_config = {}
#         self.pcolormesh_config = {}
#         self.config(**kwargs)
#
#     def config(self, logging=True, **kwargs):
#         pyclass.set_object_attributes(self, append=False, logging=logging, **kwargs)
#
#     def add_attr(self, logging=True, **kwargs):
#         pyclass.set_object_attributes(self, append=True, logging=logging, **kwargs)

#
# def to_dict(self):
#     class_vars = vars(Visual)  # get any "default" attrs defined at the class level
#     inst_vars = vars(self)  # get any attrs defined on the instance (self)
#     all_vars = dict(class_vars)
#     all_vars.update(inst_vars)
#     # filter out private attributes
#     public_vars = {k: v for k, v in all_vars.items() if not k.startswith('_')}
#     del public_vars['set_attr']
#     del public_vars['to_dict']
#     return public_vars


#
#
# class Variable(np.ndarray, npmixin.NDArrayOperatorsMixin):
#     _attrs_registered = [
#         'dataset',
#         'name', 'label', 'description', 'group',
#         'error',
#         'depends', 'unit', 'quantity_type',
#         'cs',
#         'visual'
#     ]
#
#     _dataset = None
#     _name = ''
#     _label = ''
#     _description = ''
#     _group = ''
#     _error = None
#     _unit = ''
#     _quantity_type = None
#     _cs = None
#     _depends = []
#     _visual = None
#
#     def __new__(cls, arr, copy=True, dtype=None, order='C', subok=False, ndmin=0, **kwargs):
#
#         if issubclass(arr.__class__, Variable):
#             obj_out = arr
#         else:
#             obj_out = np.array(arr, copy=copy, dtype=dtype, order=order, subok=subok, ndmin=ndmin)
#             obj_out = obj_out.view(cls)
#
#         obj_out.set_attr(**kwargs)
#         return obj_out
#
#     def __array_finalize__(self, obj):
#         # nothing needed for new variable or np.ndarray.view()
#         if obj is None or isinstance(obj.__class__, np.ndarray):
#             return None
#
#         if issubclass(obj.__class__, Variable):
#             for attr in getattr(obj, '_attrs_registered'):
#                 self.__setattr__(attr, cp.deepcopy(getattr(obj, attr)))
#         # else:
#         #     for attr in getattr(self.__class__, '_attrs_registered'):
#         #         self.__setattr__(attr, cp.deepcopy(getattr(self.__class__, attr)))
#
#     def __array_ufunc__(self, ufunc, method, *inputs, out=None, **kwargs):
#
#         args = []
#         in_no = []
#         attr_config = kwargs.pop('attr_config', None)
#
#         for i, input_ in enumerate(inputs):
#             if issubclass(input_.__class__, Variable):
#                 in_no.append(i)
#                 args.append(input_.view(np.ndarray))
#             else:
#                 args.append(input_)
#
#         outputs = out
#         out_no = []
#
#         if outputs:
#             out_args = []
#             for j, output in enumerate(outputs):
#                 if issubclass(output.__class__, Variable):
#                     out_no.append(j)
#                     out_args.append(output.view(np.ndarray))
#                 else:
#                     out_args.append(output)
#             kwargs['out'] = tuple(out_args)
#         else:
#             outputs = (None,) * ufunc.nout
#
#         results = super(self.__class__, self).__array_ufunc__(
#             ufunc, method, *args, **kwargs
#         )
#
#         if results is NotImplemented:
#             return NotImplemented
#
#         if ufunc.nout == 1:
#             results = (results,)
#
#         results = tuple((np.asarray(result).view(self.__class__)
#                          if output is None else output)
#                         for result, output in zip(results, outputs))
#
#         if len(results) == 1:
#             result = results[0]
#             result.copy_attr(self)
#             if bool(attr_config):
#                 result.set_attr(**attr_config)
#             return result
#         else:
#             return results
#
#     def __reduce__(self):
#         # patch to pickle Quantity objects (ndarray subclasses), see
#         # http://www.mail-archive.com/numpy-discussion@scipy.org/msg02446.html
#
#         object_state = list(super().__reduce__())
#         object_state[2] = (object_state[2], self.__dict__)
#         return tuple(object_state)
#
#     def __setstate__(self, state):
#         # patch to unpickle Variable objects (ndarray subclasses), see
#         # http://www.mail-archive.com/numpy-discussion@scipy.org/msg02446.html
#
#         nd_state, own_state = state
#         super().__setstate__(nd_state)
#         self.__dict__.update(own_state)
#
#     def __repr__(self):
#         str_pref = '<Geospace ' + self.__class__.__name__ + ' '
#         str_arr = np.array2string(self.view(np.ndarray), separator=', ',
#                                   prefix=str_pref)
#         return f'{str_pref}{str_arr}{self.unit:s}>'
#
#     def config(self, **kwargs):
#         self.set_attr(**kwargs)
#
#     def set_attr(self, **kwargs):
#         # set values for the registered attributes
#         attr_add = kwargs.pop('attr_add', True)
#         logging = kwargs.pop('logging', False)
#         new_attrs = []
#         for key, value in kwargs.items():
#             if key in self._attrs_registered:
#                 setattr(self, key, value)
#             elif attr_add:
#                 setattr(self, key, value)
#                 self._attrs_registered.append(key)
#                 new_attrs.append(key)
#
#         if logging:
#             if attr_add:
#                 logger.StreamLogger.info('Attrs: {} added!'.format(', '.join(new_attrs)))
#             else:
#                 logger.StreamLogger.warning('Attrs: {} not added!'.format(', '.join(new_attrs)))
#
#     def copy_attr(self, obj):
#         for attr in getattr(obj, '_attrs_registered'):
#             self.__setattr__(attr, cp.deepcopy(getattr(obj, attr)))
#
#     @property
#     def dataset(self):
#         return self._dataset
#
#     @dataset.setter
#     def dataset(self, obj):
#         from geospacelab.datahub import Dataset
#         # Check the type:
#         if obj is None:
#             self._dataset = None
#         elif not isinstance(obj, Dataset):
#             raise ValueError
#         self._dataset = obj
#
#     @property
#     def name(self):
#         return self._name
#
#     @name.setter
#     def name(self, str_):
#         # Check the type:
#         if str_ is None:
#             str_ = ''
#         elif not isinstance(str_, str):
#             raise ValueError
#         self._name = str_
#
#     @property
#     def label(self):
#         return self._label
#
#     @label.setter
#     def label(self, str_):
#         # Check the type:
#         if str_ is None:
#             str_ = ''
#         elif not isinstance(str_, str):
#             raise ValueError
#         self._label = str_
#
#     @property
#     def description(self):
#         return self._description
#
#     @description.setter
#     def description(self, str_):
#         # Check the type:
#         if str_ is None:
#             str_ = ''
#         elif not isinstance(str_, str):
#             raise ValueError
#         self._description = str_
#
#     @property
#     def group(self):
#         return self._group
#
#     @group.setter
#     def group(self, str_):
#         # Check the type:
#         if str_ is None:
#             str_ = ''
#         elif not isinstance(str_, str):
#             raise ValueError
#         self._group = str_
#
#     @property
#     def error(self):
#         return self._error
#
#     @error.setter
#     def error(self, err):
#         if err is None:
#             self._error = None
#         if isinstance(err, self.__class__):
#             self._error = err
#         elif isinstance(err, str):
#             self._error = self.dataset[err]
#
#     @property
#     def unit(self):
#         return self._unit
#
#     @unit.setter
#     def unit(self, value):
#         # Check the type:
#         if value is None:
#             value = ''
#         elif not isinstance(value, str):
#             raise ValueError
#         self._unit = value
#
#     @property
#     def quantity_type(self):
#         return self._quantity_type
#
#     @quantity_type.setter
#     def quantity_type(self, str_):
#         # Check the type:
#         if str_ is None:
#             str_ = ''
#         elif not isinstance(str_, str):
#             raise ValueError
#         self._quantity_type = str_
#
#     @property
#     def cs(self):
#         return self._cs
#
#     @cs.setter
#     def cs(self, obj):
#         # Check the type:
#         self._cs = obj
#
#     @property
#     def depends(self):
#         return self._depends
#
#     @depends.setter
#     def depends(self, value):
#         # Check the type:
#         if value is None:
#             value = []
#         elif not isinstance(value, list):
#             raise ValueError
#         self._depends = value
#
#     @property
#     def visual(self):
#         return self._visual
#
#     @visual.setter
#     def visual(self, obj):
#         # Check the type:
#         if obj is None:
#             self._visual = None
#         elif not isinstance(obj, Visual):
#             raise ValueError
#         self._visual = obj
#
#
# if __name__ == "__main__":
#     a = Variable([1, 2, 3, 4], name='a')
#     a = a.reshape((a.shape[0], 1))
#     b = 5 + a
#     c = np.sum(a)
#
#     pass
