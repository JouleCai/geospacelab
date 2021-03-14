
"""Dataset and variable models"""
import weakref
import numpy as np
import copy

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

class GeospaceVariable(np.ndarray):
    """ Set up geospace variables  
    :date:  2021-03-13

    To be implemented.
    """
    def __new__(cls, input_array, **kwargs):
    # Input array is a list, truple, or np.ndarray, or SpPhyVariable instance
        if isinstance(input_array, GeospaceVariable):
            obj = input_array
        else:
            # The input array is casted to the SpPHyVariable type, the default copy is False
            copy_arr = kwargs.pop('copy', False)
            obj = np.array(input_array, copy=copy_arr)
            obj = obj.view(cls)
            # add the new attributes to the created instance
            obj.attributes = GeospaceVariableAttribute(obj, **kwargs)
        # Finally, we must return the newly created object:
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.attributes = copy.deepcopy(getattr(obj, 'attributes', None))  # does deep copy need?

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
    

class GeospaceVariableAttribute(object):
    """ ISTP/IAGG styled the attributes for GeospaceVariables.
    :Date: 2021-03-13

    The geospace variable attributes  Some of the arributes are set by @property decorator.

    -  **parameterss** and **return**
        :param dataset: parent dataset
        :param datatype: 'data', 'support_data', 'metadata'
        :param field: 'scalar', 'vector', or 'tensor'
        :param name: variable name, use symbols
        :param plainname: variable plain name, use descriptive text
        :param label: latex label if any
        :param error: error of the variable stored in parent dataset
        :param unit: variable unit
        :param dim: dimesion of the variable 0 for 'metadata' and 1,2,3... for 'data' or 'support_data' 
        :param depend: dependencies at each dimesional axis.
        :param cs: coordinate system if any
        :param group: variable group if any
        :param base_phy_quatity: base physical quantity object
        :param visual: visual object for plotting

    - **Example**
        :Example:
        
    """
    def __init__(self, var_obj, **kwargs):
        """ Set default values
        """
        self.dataset = kwargs.pop('dataset', None)

        self.field = kwargs.pop('field', 'scalar') # ['scalar']   # Variable types can be 'scalar', 'vector', 'tensor'
        self.name = kwargs.pop('name', None)
        self.plainname = kwargs.pop('plainname', None)
        self.label = kwargs.pop('label', None)
        self.group = kwargs.pop('group', None)

        self.error = kwargs.pop('error', None)
        self.depends = kwargs.pop('depends', [])  # list, dimensional dependencies

        self.unit = kwargs.pop('unit', None)

        self.cs = kwargs.pop('cs', None)

        no_visual = kwargs.pop('no_visual', False)

        if not no_visual:
            config_visual = kwargs.pop('config_visual', {})
            self.visual = Visual(self, **config_visual)

        self.base_phy_quantity = kwargs.pop('base_phy_quantity', None)

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
            return GeospaceVariable(self._error)

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