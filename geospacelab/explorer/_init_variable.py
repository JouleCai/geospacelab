
"""Variable model based on np.ndarray"""
import weakref
import numpy as np
from numpy.lib.arraysetops import isin
import numpy.lib.mixins as npmixin
import copy

import geospacelab.utilities.logging_config as mylog

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
        'depends'
        ]

    def __new__(cls, array_in, **kwargs):
    # Input array is a list, truple, or np.ndarray, or SpPhyVariable instance
        if isinstance(array_in, BaseVariable):
            obj = array_in
        else:
            # The input array is casted to the SpPHyVariable type, the default copy is False
            copy_arr = kwargs.pop('copy', False)
            obj = np.array(array_in, copy=copy_arr)
            obj = obj.view(cls)
            # add the new attributes to the created instance
            obj._init_attribues(obj, **kwargs)
        # Finally, we must return the newly created object:
        return obj
    
    # set attributes for the variable
    def _init_attributes(self, **kwargs):
        
        # set default values for the registered attributes
        for key in self._attributes_registered:
            setattr(self, key, kwargs.pop(key, None))

        # set the attributes without registration
        attr_add = kwargs.pop('attr_add', {})
        if bool(attr_add):
            self.add_attributes(**kwargs)
        return

    def __array_finalize__(self, obj):
        if obj is None:
            return
        if issubclass(obj, BaseVariable):
            self.copy_attrbutes(obj)
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
            if issubclass(input_, BaseVariable):
                in_no.append(i)
                args.append(input_.view(np.ndarray))
            else:
                args.append(input_)

        outputs = out
        out_no = []

        if outputs:
            out_args = []
            for j, output in enumerate(outputs):
                if issubclass(output, BaseVariable):
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
            attrs = obj._attributes_regitered
        for attr in attrs:
            if force:
                self.add_attributes({attr: getattr(obj, attr)})
            else:
                if attr in self._attributes_registered:
                    setattr(self, attr, getattr(obj, attr))
            
    @property
    def dataset(self):
        if self._dataset_ref is None:
            print("The attribute dataset has not been assigned!")
            return None
        return self._dataset_ref()

    @dataset.setter
    def dataset(self, obj):
        # Check the type:
        if obj is None:
            self._dataset_ref = None
        elif isinstance(obj, Dataset):
            self._dataset_ref = weakref.ref(obj)
        else:
            raise AttributeError('The attribute "dataset" must be a Dataset object or None!')

    @property
    def error(self):
        if self._error is None:
            return None
        elif isinstance(self._error, str):
            return self.dataset[self._error]
        else:
            return BaseVariable(self._error)

    @error.setter
    def error(self, err):
        self._error = err

    @property
    def depends(self):
        return self._depends

    @depends.setter
    def depends(self, dps):
        if not isinstance(dps, list):
            raise ValueError('The attribute "Depends" must be a list!')
        self._depends = dps

    @property
    def label(self):
        return self._label

    @label.setter
    def label(self, str1):
        if str1 is None:
            self._label = self.name
        else:
            self._label = r'{}'.format(self.str1)


class Visual(object):
    def __init__(self, attrs_obj, **kwargs):
        self.plottype = kwargs.pop('plottype', None)
        self.xdata = kwargs.pop('xdata', None)
        self.ydata = kwargs.pop('ydata', None)
        self.zdata = kwargs.pop('zdata', None)
        self.xdata_scale = kwargs.pop('xdata_scale', 1.)
        self.ydata_scale = kwargs.pop('ydata_scale', 1.)
        self.zdata_scale = kwargs.pop('zdata_scale', 1.)
        self.xdata_res = kwargs.pop('xdata_res', None)
        self.ydata_res = kwargs.pop('ydata_res', None)
        self.zdata_res = kwargs.pop('zdata_res', None)
        self.xdata_err = kwargs.pop('xdata_err', None)
        self.ydata_err = kwargs.pop('ydata_err', None)
        self.zdata_err = kwargs.pop('zdata_err', None)
        # self.xdata_mask = kwargs.pop('xdata_mask', None)
        # self.ydata_mask = kwargs.pop('ydata_mask', None)
        # self.zdata_mask = kwargs.pop('zdata_mask', None)
        self.xaxis_lim = kwargs.pop('xaxis_lim', None)
        self.yaxis_lim = kwargs.pop('yaxis_lim', None)
        self.zaxis_lim = kwargs.pop('zaxis_lim', None)
        self.xaxis_label = kwargs.pop('xaxis_label', None)
        self.yaxis_label = kwargs.pop('yaxis_label', None)
        self.zaxis_label = kwargs.pop('zaxis_label', None)
        self.xaxis_scale = kwargs.pop('xaxis_scale', None)
        self.yaxis_scale = kwargs.pop('yaxis_scale', None)
        self.zaxis_scale = kwargs.pop('zaxis_scale', None)
        self.xdata_unit = kwargs.pop('xdata_unit', None)
        self.ydata_unit = kwargs.pop('ydata_unit', None)
        self.zdata_unit = kwargs.pop('zdata_unit', None)
        self.xaxis_ticks = kwargs.pop('xaxis_ticks', None)
        self.yaxis_ticks = kwargs.pop('yaxis_ticks', None)
        self.zaxis_ticks = kwargs.pop('zaxis_ticks', None)
        self.xaxis_ticklabels = kwargs.pop('xaxis_ticklabels', None)
        self.yaxis_ticklabels = kwargs.pop('yaxis_ticklabels', None)
        self.zaxis_ticklabels = kwargs.pop('zaxis_ticklabels', None)

        self.colormap = kwargs.pop('colormap', None)
        self.visible = kwargs.pop('visible', True)
        self.kwargs_plot = kwargs.pop('kwargs_draw', {})

        # self.set_attr(**kwargs)

    @property
    def plottype(self):
        return self._plottype

    @plottype.setter
    def plottype(self, code):
        attrs_obj = self._attrs_obj_ref()
        var_obj = attrs_obj.variable
        ndim = var_obj.ndim
        if code is None:
            if attrs_obj.type == 'scalar':
                self._plottype = str(ndim) + 'D'
            elif attrs_obj.type == 'vector':
                self._plottype = str(ndim - 1) + 'D'
            elif attrs_obj.type == 'tensor':    # for future extension.
                self._plottype = str(ndim - 2) + 'D'
        else:
            self._plottype = code


class Dataset(object):
    pass