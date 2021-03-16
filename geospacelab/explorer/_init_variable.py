"""Variable model based on np.ndarray"""

import weakref

import copy
import numpy as np
import numpy.lib.mixins as npmixin
from geospacelab.explorer._init_dataset import Dataset
from geospacelab.toolbox import logger
from geospacelab.toolbox.graphic import Visual

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


class Variable(np.ndarray, npmixin.NDArrayOperatorsMixin):
    
    _attrs_registered = [
        'dataset', 
        'name', 'label', 'description', 'group', 
        'error', 
        'depends', 'unit', 'quantity_type',
        'cs',
        'visual'
    ]

    _dataset = None
    _name = ''
    _label = ''
    _description = ''
    _group = ''
    _error = None
    _unit = ''
    _quantity_type = None
    _cs = None
    _depends = []
    _visual = None

    def __new__(cls, arr, copy=True, dtype=None, order='C', subok=False, ndmin=0, **kwargs):
        if isinstance(arr, BaseVariable):
            obj_out = arr
        else:
            obj_out = np.array(arr, copy=copy, dtype=dtype, order=order, subok=subok, ndmin=ndmin)
            obj_out = obj_out.view(cls)

        obj_out.set_attr(**kwargs)

        return obj_out

    def __array_finalize__(self, obj):
        if obj is None:
            return None
        
        if issubclass(obj.__class__, Variable):
            for attr in obj._attrs_registered:
                self.__setattr__(attr, getattr(obj, attr, None))
        

    def set_attr(self, **kwargs):
        # set values for the registered attributes
        attr_add = kwargs.pop('attr_add', True) 
        logging = kwargs.pop('logging', False)
        new_attrs = []
        for key, value in kwargs.items():
            if key in self._attributes_registered:
                setattr(self, key, value)
            elif attr_add:
                setattr(self, key, value)
                self._attrs_registered.append(key)
                new_attrs.append(key)

        if logging:
            if attr_add:
                logger.StreamLogger.info('Attrs: {} added!'.format(', '.join(new_attrs)))
            else:
                logger.warning('Attrs: {} not added!'.format(', '.join(new_attrs)))

    def copy_attr(obj):
        pass


    @property
    def dataset(self):
        return self._dataset
    
    @dataset.setter
    def dataset(self, obj):
        # Check the type:
        if obj is None:
            self._dataset = None
        elif not isinstance(obj, Dataset):
            return ValueError
        self._dataset = obj

    
    @property
    def name(self):
        return self._name
    
    @name.setter
    def name(self, str_):
        # Check the type:
        if str_ is None:
            str_ = ''
        elif not isinstance(str_, str):
            return ValueError
        self._name = str_
    
    @property
    def label(self):
        return self._label
    
    @label.setter
    def label(self, str_):
        # Check the type:
        if str_ is None:
            str_ = ''
        elif not isinstance(str_, str):
            return ValueError
        self._label = str_

    @property
    def description(self):
        return self._description
    
    @description.setter
    def description(self, str_):
        # Check the type:
        if str_ is None:
            str_ = ''
        elif not isinstance(str_, str):
            return ValueError
        self._description = str_
    
    @property
    def group(self):
        return self._group
    
    @group.setter
    def group(self, str_):
        # Check the type:
        if str_ is None:
            str_ = ''
        elif not isinstance(str_, str):
            return ValueError
        self._group = str_
    
    @property
    def error(self):
        return self._error
    
    @error.setter
    def error(self, err):
        if err is None:
            self._error = None
        if isinstance(err, self.__class__):
            self._error = err
        elif isinstance(err, str):
            self._error = self.dataset[err]

    @property
    def unit(self):
        return self._unit
    
    @unit.setter
    def unit(self, value):
        # Check the type:
        if value is None:
            value = ''
        elif not isinstance(value, str):
            return ValueError
        self._unit = value

    @property
    def quantity_type(self):
        return self._quantity_type
    
    @quantity_type.setter
    def quantity_type(self, str_):
        # Check the type:
        if str_ is None:
            str_ = ''
        elif not isinstance(str_, str):
            return ValueError
        self._quantity_type = str_

    @property
    def cs(self):
        return self._cs
    
    @cs.setter
    def cs(self, obj):
        # Check the type:
        self._cs = obj

    @property
    def depends(self):
        return self._depends
    
    @depends.setter
    def depends(self, value):
        # Check the type:
        if value is None:
            value = []
        elif not isinstance(value, list):
            return ValueError
        self._depends = value

    @property
    def visual(self):
        return self._group
    
    @visual.setter
    def visual(self, obj):
        # Check the type:
        if obj is None:
            obj = None
        elif not isinstance(obj, Visual):
            return ValueError
        self._visual = obj


class BaseVariable(np.ndarray, npmixin.NDArrayOperatorsMixin):
    """ Set up geospace variables  
    :date:  2021-03-14

    To be implemented.
    """
    _attrs_registered = ['unit', 'error', 'depends', 'dataset', 'name', 'type', 'coords', 'visual']
    _dataset = None
    _name = ''
    _label = ''
    _description = ''
    _group = ''
    _error = None
    _unit = ''
    _quantity_type = None
    _cs = None
    _depends = []
    _visual = None

    def __new__(cls, arr, copy=True, dtype=None, order='C', subok=False, ndmin=0, **kwargs):
        if isinstance(arr, BaseVariable):
            obj_out = arr
        else:
            obj_out = np.array(arr, copy=copy, dtype=dtype, order=order, subok=subok, ndmin=ndmin)
            obj_out = obj_out.view(cls)

        obj_out._extra_attrs = kwargs.pop('extra_attrs', False)
        obj_out._attr_register(**kwargs)

        return obj_out
    
    # set attributes for the variable
    def _init_attributes(self, **kwargs):
        # set values for the registered attributes
        attr_add = kwargs.pop('attr_add', True) 
        for key, value in kwargs.items():
            if key in self._attributes_registered:
                setattr(self, key, value)
            elif attr_add:
                self.add_attributes(**{key: value})
            else:
                logger.warning("Attr: {} was not added!".format(key))
                
    def __array_finalize__(self, obj):
        if obj is None:
            return None
        for attr in self._attributes_registered:
            self.__setattr__(attr, copy.deepcopy(getattr(obj, attr, {})))

    def __array_ufunc__(self, ufunc, method, *inputs, out=None, **kwargs):

        args = []
        in_no = []
        attr_config = kwargs.pop('attr_config', None)

        for i, input_ in enumerate(inputs):
            if issubclass(input_.__class__, BaseVariable):
                in_no.append(i)
                args.append(input_.view(np.ndarray))
            else:
                args.append(input_)

        outputs = out
        out_no = []

        if outputs:
            out_args = []
            for j, output in enumerate(outputs):
                if issubclass(output.__class__, BaseVariable):
                    out_no.append(j)
                    out_args.append(output.view(np.ndarray))
                else:
                    out_args.append(output)
            kwargs['out'] = tuple(out_args)
        else:
            outputs = (None,) * ufunc.nout
        
        results = super(self.__class__,  self).__array_ufunc__(
            ufunc, method, *args, **kwargs
        )

        if results is NotImplemented:
            return NotImplemented

        if ufunc.nout == 1:
            results = (results,)
        a = np.asarray(results[0]) 
        test = a.view(self.__class__)
        results = tuple((np.asarray(result).view(self.__class__)
                         if output is None else output)
                        for result, output in zip(results, outputs))
        
        if len(results) == 1:
            result = results[0]
            result.copy_attributes(self)
            if bool(attr_config):
                result.set_attributes(attr_config)
        else:
            return results
        
    def add_attributes(self, **kwargs):
        # add attributes to object
        for key, value in kwargs.items():
            self.__setattr__(key, value)
            self._attributes_registered.append(key)

    def copy_attributes(self, obj, force=True, required_attributes=None):
        # copy attributes from obj to self
        if required_attributes is not None:
            attrs = required_attributes
        else:
            attrs = obj._attributes_registered
        for attr in attrs:
            if force:
                self.add_attributes(**{attr: getattr(obj, attr)})
            else:
                if attr in self._attributes_registered:
                    self.__setattr__(attr, getattr(obj, attr))
            
    @property
    def dataset(self):
        return self._dataset
    
    @dataset.setter
    def dataset(self, obj):
        # Check the type:
        if not isinstance(obj, Dataset):
            return ValueError
        self._dataset = obj
    
    @property
    def name(self):
        return self._name
    
    @name.setter
    def name(self, str_):
        # Check the type:
        if not isinstance(str_, str):
            return ValueError
        self._name = str_
    
    @property
    def label(self):
        return self._label
    
    @label.setter
    def label(self, str_):
        # Check the type:
        if not isinstance(str_, str):
            return ValueError
        self._label = str_

    @property
    def description(self):
        return self._description
    
    @description.setter
    def description(self, str_):
        # Check the type:
        if not isinstance(str_, str):
            return ValueError
        self._description = str_
    
    @property
    def group(self):
        return self._group
    
    @group.setter
    def group(self, str_):
        # Check the type:
        if not isinstance(str_, str):
            return ValueError
        self._group = str_
    
    @property
    def error(self):
        return self._error
    
    @error.setter
    def error(self, err):
        if isinstance(err, self.__class__):
            self._error = err
        elif isinstance(err, str):
            self._error = self.dataset[err]
        else:
            return ValueError

    @property
    def unit(self):
        return self._unit
    
    @unit.setter
    def unit(self, value):
        # Check the type:
        if not isinstance(value, str):
            return ValueError
        self._unit = value

    @property
    def quantity_type(self):
        return self._quantity_type
    
    @quantity_type.setter
    def quantity_type(self, str_):
        # Check the type:
        if not isinstance(str_, str):
            return ValueError
        self._quantity_type = str_

    @property
    def cs(self):
        return self._cs
    
    @cs.setter
    def group(self, obj):
        # Check the type:
        self._cs = obj

    @property
    def depends(self):
        return self._depends
    
    @depends.setter
    def depends(self, value):
        # Check the type:
        if not isinstance(value, list):
            return ValueError
        self._depends = value

    @property
    def visual(self):
        return self._group
    
    @visual.setter
    def visual(self, obj):
        # Check the type:
        if not isinstance(obj, Visual):
            return ValueError
        self._visual = obj



if __name__ == "__main__":
    a = Variable([1, 2, 3, 4])
    b = a + 5
    pass