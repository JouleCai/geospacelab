"""Variable model based on np.ndarray"""

import weakref
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


class BaseVariable(np.ndarray, npmixin.NDArrayOperatorsMixin):
    """ Set up geospace variables  
    :date:  2021-03-14

    To be implemented.
    """
    _attributes_registered = [
        'dataset', 
        'name', 'label', 'description', 'group', 
        'error',
        'unit', 'quantity_type', 'cs',
        'depends',
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

    def __new__(cls, array_in, **kwargs):
        # Input array is a list, truple, or np.ndarray, or SpPhyVariable instance
        if isinstance(array_in, BaseVariable):
            obj = array_in
        else:
            # The input array is casted to the SpPHyVariable type, the default copy is False
            # copy_arr = kwargs.pop('copy', False)
            obj = np.asarray(array_in).view(cls)
            # add the new attributes to the created instance
            obj._init_attributes(**kwargs)
        # Finally, we must return the newly created object:
        return obj
    
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
            return
        if issubclass(obj.__class__, BaseVariable):
            self.copy_attributes(obj)
        else:
            return

    def __reduce__(self):
        """
        This is called when pickling, see:
        http://www.mail-archive.com/numpy-discussion@scipy.org/msg02446.html
        for this particular example.
        """
        object_state = list(np.ndarray.__reduce__(self))
        subclass_state = (self.attributes,)
        object_state[2] = (object_state[2], subclass_state)
        return tuple(object_state)

    def __setstate__(self, state):
        """
        Used for unpickling after __reduce__
        """
        nd_state, own_state = state
        np.ndarray.__setstate__(self, nd_state)

        info, = own_state
        self.attributes = info

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
            setattr(self, key, value)
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
                    setattr(self, attr, getattr(obj, attr))
            
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
    a = BaseVariable([1, 2, 3, 4])
    b = a + 5
    pass