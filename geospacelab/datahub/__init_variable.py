"""Variable model based on np.ndarray"""


__author__ = "Lei Cai"
__copyright__ = "Copyright 2021, The GeoSpaceLab Project"
__credits__ = ["Lei Cai"]
__license__ = "MLT"
# __version__ = "1.0.1"
__maintainer__ = "Lei Cai"
__email__ = "lei.cai@oulu.fi"
__status__ = "Developing"
__revision__ = ""
__docformat__ = "reStructureText"

import copy
import weakref

import numpy as np
import geospacelab.toolbox.utilities.pyclass as pyclass
import geospacelab.toolbox.utilities.pylogging as mylog


class VariableModel(object):

    def __init__(self, **kwargs):
        self.name = kwargs.pop('name', '')
        self.fullname = kwargs.pop('fullname', '')

        self.label = kwargs.pop('label', '')

        self.category = kwargs.pop('category', '')  # 'support_data', 'data', 'metadata'
        self.group = kwargs.pop('group', '')

        self.unit = kwargs.pop('unit', '')
        self.unit_label = kwargs.pop('unit_label', '')

        self.quantity = kwargs.pop('quantity', None)

        self.value = kwargs.pop('value', None)
        self.error = kwargs.pop('error', None)

        self.variable_type = kwargs.pop('variable_type', None)    # scalar, vector, tensor, ...
        self.ndim = kwargs.pop('ndim', None)
        self.depends = kwargs.pop('depends', {})

        self.dataset = kwargs.pop('dataset', None)

        self.visual = None

        visual = kwargs.pop('visual', 'off')
        if visual == 'on':
            self.visual = Visual()
            self.visual.config(**kwargs.pop('visual_config', {}))

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
                        value = self.dataset[value]
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

    def get_visual_attr(self, attr_name):
        value = getattr(self.visual, attr_name)
        if isinstance(value, tuple):
            try:
                value = getattr(self, value[0])
            except AttributeError:
                value_new = []
                for elem in value:
                    value_new.append(self.dataset[elem])
                value = value_new
        return value

    @property
    def dataset(self):
        return self._dataset_proxy

    @dataset.setter
    def dataset(self, dataset_obj):
        if dataset_obj is not None:
            self._dataset_proxy = weakref.proxy(dataset_obj)

    @property
    def value(self):
        v = None
        if isinstance(self._value, str):
            v = self.dataset[self._value]
        else:
            v = self._value
        if v is None:
            mylog.StreamLogger.warning("The variable ({})'s value has not been assigned!".format(self.name))
        return v

    @value.setter
    def value(self, v):
        if isinstance(v, list):
            v = np.array(v)
        self._value = v

    @property
    def error(self):
        v = None
        if isinstance(self._error, str):
            v = self.dataset[self._error]
        else:
            v = self._error
        if v is None:
            mylog.StreamLogger.warning("The variable ({})'s error has not been assigned!".format(self.name))
        return v

    @error.setter
    def error(self, v):
        if isinstance(v, list):
            v = np.array(v)
        self._error = v

    @property
    def unit_label(self):
        label = self._unit_label
        if label is None:
            label = self.unit
        return label

    @unit_label.setter
    def unit_label(self, label):
        self._unit_label = label

    @property
    def label(self):
        lb = self._label
        if lb is None:
            lb = self.name
        return lb

    @label.setter
    def label(self, lb):
        self._label = lb

    @property
    def variable_type(self):
        return self._variable_type

    @variable_type.setter
    def variable_type(self, value):
        if value is None:
            self._variable_type = None
        elif value.lower() not in ['scalar', 'vector', 'matrix']:
            raise ValueError
        else:
            self._variable_type = value.lower()

    @property
    def ndim(self):
        if self._ndim is None:
            var_type = self.variable_type
            if var_type is None:
                # mylog.StreamLogger.warning('the variable type has not been defined! Use default type "Scalar"...')
                var_type = "scalar"
            value = self.value
            if value is None:
                mylog.StreamLogger.warning("return None!")
                return None
            shape = value.shape
            if var_type == 'scalar':
                offset = 0
            if var_type == 'vector':
                offset = -1
            if var_type == 'matrix':
                offset = -2
            if len(shape) == 2 and shape[1] == 1:
                nd = 1
            else:
                nd = len(shape) - offset
            if nd <= 0:
                raise ValueError("ndim cannot be a non-positive integer")
        return nd

    @ndim.setter
    def ndim(self, value):
        if value is not None and type(value) is not int:
            raise TypeError
        self._ndim = value

    @property
    def depends(self):
        return self._depends

    @depends.setter
    def depends(self, d_dict):
        self._depends = {}
        if not dict(d_dict):
            self._depends = {}
        else:
            self._depends = {}
            for key, value in d_dict.items():
                if type(key) is not int:
                    raise KeyError
                if value is None:
                    self._depends[key] = {}
                else:
                    self._depends[key] = value


class Visual_Data:
    def __init__(self):
        self.data = None
        self.error = None
        self.scale = 1.
        self.res = None     # resolution


class Axis:
    def __init__(self):
        self.label = None
        self.unit = None
        self.limit = None
        self.invert = False
        self.ticks = None
        self.tick_labels = None
        self.visible = True


class Plot_Config:
    def __init__(self):
        self.color = None
        self.visible = True
        self.line = {}
        self.errorbar = {}
        self.pcolormesh = {}
        self.imshow = {}
        self.scatter = {}
        self.mask_gap = True


class Visual_new(object):
    _data = {}
    _axis = {}
    _variable_proxy = VariableModel()

    def __init__(self, **kwargs):
        self.variable = kwargs.pop('variable', None)
        self.data = None
        self.axis = None
        self.plot = None

    @property
    def variable(self):
        return self._variable_proxy

    @variable.setter
    def variable(self, var_obj):
        if var_obj is not None:
            self._variable_proxy = weakref.proxy(var_obj)

    @property
    def data(self):
        if not dict(self._data):
            try:
                for ind in range(self.variable.ndim + 1):
                    self._data[ind] = Visual_Data()
            except:
                self._data = {}
        return self._data

    @data.setter
    def data(self, d_dict):
        if dict(d_dict):
            self._data = d_dict
        elif d_dict is not None:
            raise TypeError

    @property
    def axis(self):
        if not dict(self._axis):
            try:
                for ind in range(self.variable.ndim + 1):
                    self._axis[ind] = Visual_Data()
            except:
                self._axis = {}
        return self._axis

    @axis.setter
    def axis(self, a_dict):
        if dict(a_dict):
            self._axis = a_dict
        elif a_dict is not None:
            raise TypeError


class Visual(object):
    def __init__(self, **kwargs):
        self.plot_type = None
        self.data_0 = None
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
#         'coords',
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
