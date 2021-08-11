""" Model classes for geospace variables, e.g., VariableModel, Visual, ...
"""


import copy
import weakref
import re

import numpy as np
import geospacelab.toolbox.utilities.pyclass as pyclass
import geospacelab.toolbox.utilities.pylogging as mylog
import geospacelab.toolbox.utilities.pybasic as basic


class VariableModel(object):
    """VariableModel is a base class for a geospace variable with useful attributes
    
    :param name: The variable's name, ['']
    :type name: str
    :param fullname: The variable's full name, ['']
    :type fullname: str
    :param label: The variable's label for display. If a raw string (e.g., r'$\alpha$'),
        it will show a latex format.
    :type label: str
    :param data_type: The variable's data type in one of them: 'support_data', 'data', and 'metadata'
        in the same format as in a NASA's cdf file.
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
        self._name = ''
        self._fullname = ''
        self._label = ''
        self._data_type = None
        self._group = ''
        self._unit = ''
        self._unit_label = ''
        self._quantity = None

        self._value = None
        self._error = None
        self._variable_type = None

        self._ndim = None
        self._depends = {}

        self._dataset_proxy = None

        self._visual = None

        self.name = kwargs.pop('name', self._name)
        self.fullname = kwargs.pop('fullname', self._fullname)

        self.label = kwargs.pop('label', self._label)

        self.data_type = kwargs.pop('data_type', self._data_type)  # 'support_data', 'data', 'metadata'
        self.group = kwargs.pop('group', self._group)

        self.unit = kwargs.pop('unit', self._unit)
        self.unit_label = kwargs.pop('unit_label', self._unit_label)

        self.quantity = kwargs.pop('quantity', self._quantity)

        self.value = kwargs.pop('value', self._value)
        self.error = kwargs.pop('error', self._error)

        self.variable_type = kwargs.pop('variable_type', self._variable_type)    # scalar, vector, tensor, ...
        self.ndim = kwargs.pop('ndim', self._ndim)
        self.depends = kwargs.pop('depends', self._depends)

        self.dataset = kwargs.pop('dataset', None)

        self.visual = kwargs.pop('visual', 'off')

    def config(self, logging=True, **kwargs):
        pyclass.set_object_attributes(self, append=False, logging=logging, **kwargs)

    def add_attr(self, logging=True, **kwargs):
        pyclass.set_object_attributes(self, append=True, logging=logging, **kwargs)

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
                depend_new[key] = value
        return depend_new

    def set_depend(self, axis, depend_dict):
        if not isinstance(depend_dict, dict):
            raise TypeError
        self.depends[axis] = depend_dict

    # def get_visual_data(self, attr_name):
    #     data = []
    #     if 'data' not in attr_name:
    #         raise AttributeError("Input must be a data attribute in the visual instance!")
    #
    #     attr = getattr(self.visual, attr_name)
    #     axis = ord('z') - ord(attr[0])
    #     if attr is None:
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
            results.append(result)
        if type_attr is not list:
            results = results[0]
        return results

    def join(self, var_new):
        if issubclass(var_new.__class__, VariableModel):
            value = var_new.value
        else:
            value = var_new

        if self.value is None:
            self.value = value
        else:
            self.value = np.concatenate((self.value, var_new), axis=0)

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
        return self._dataset_proxy

    @dataset.setter
    def dataset(self, dataset_obj):
        if dataset_obj is None:
            return

        from geospacelab.datahub._dataset_base import DatasetModel
        if issubclass(dataset_obj.__class__, DatasetModel):
            self._dataset_proxy = weakref.proxy(dataset_obj)
        else:
            raise TypeError

    @property
    def value(self):
        v = None
        if isinstance(self._value, str):
            v = self.dataset[self._value].value
        else:
            v = self._value
        return v

    @value.setter
    def value(self, v):
        # type check
        if v is None:
            return
        if type(v) is str:
            self._value = v
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
        self._value = v

    @property
    def error(self):
        v = None
        if isinstance(self._error, str):
            v = self.dataset[self._error].value
        else:
            v = self._error
        return v

    @error.setter
    def error(self, v):
        if v is None:
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
        if type(lb) is not str:
            raise TypeError
        self._label = lb

    @property
    def variable_type(self):
        return self._variable_type

    @variable_type.setter
    def variable_type(self, value):
        if value is None:
            return
        if value.lower() not in ['scalar', 'vector', 'matrix', 'tensor']:
            raise AttributeError
        self._variable_type = value.lower()

    @property
    def ndim(self):
        if self._ndim is None:
            var_type = self.variable_type
            if var_type is None:
                mylog.StreamLogger.warning('the variable type has not been defined! Use default type "Scalar"...')
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
            return
        if type(value) is not int:
            raise TypeError
        self._ndim = value

    @property
    def depends(self):
        return self._depends

    @depends.setter
    def depends(self, d_dict):
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
        self.label = None
        self.label_style = 'double'   # or 'single
        self.label_pos = None       # label position
        self.unit = None
        self.lim = None
        self.scale = 'linear'
        self.invert = False
        self.ticks = None
        self.tick_labels = None
        self.minor_ticks = None
        self.major_tick_max = 6
        self.minor_tick_max = 30
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
        self._variable_proxy = None
        self._ndim = None

        self.variable = kwargs.pop('variable', None)
        self.axis = kwargs.pop('axis', None)
        self.plot_config = kwargs.pop('plot_config', VisualPlotConfig())
        self.ndim = kwargs.pop('ndim', self._ndim)

    def config(self, logging=True, **kwargs):
        pyclass.set_object_attributes(self, append=False, logging=logging, **kwargs)

    def add_attr(self, logging=True, **kwargs):
        pyclass.set_object_attributes(self, append=True, logging=logging, **kwargs)

    @property
    def variable(self):
        return self._variable_proxy

    @variable.setter
    def variable(self, var_obj):
        if var_obj is None:
            return
        if issubclass(var_obj.__class__, VariableModel):
            self._variable_proxy = weakref.proxy(var_obj)
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
