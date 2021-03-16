
from geospacelab.dataexplorer._init_variable import Variable

import geospacelab.toolbox.utilities.pyclass as myclass
import geospacelab.utilities.pybasic as mybasic


# BaseClass with the "set_attr" method
class BaseClass(object):
    def __init__(self, **kwargs):
        self.name = kwargs.pop('name', None)
        self.category = kwargs.pop('category', None)
        self.label = None

    def label(self, fields=None, fields_ignore=None, separator='_', lowercase=True):
        if fields_ignore is None:
            fields_ignore = ['category', 'note']
        sublabels = []
        if fields is None:
            attrs = myclass.get_object_attributes(self)
            for attr, value in attrs.items():
                if attr in fields_ignore:
                    continue
                if not isinstance(attr, str):
                    continue
                sublabels.append(value)
        else:
            for field in fields:
                sublabels.append(getattr(self, field))
        label = mybasic.str_join(sublabels, separator=separator, lowercase=lowercase)
        return label

    def set_attr(self, **kwargs):
        append = kwargs.pop('append', True)
        logging = kwargs.pop('logging', False)
        myclass.set_object_attributes(self, append=append, logging=logging, **kwargs)


# Class Database
class Database(BaseClass):
    def __init__(self, name='temporary', category='local', **kwargs):
        self.name = name
        self.category = category
        super().__init__(name=self.name, category=self.category)
        self.set_attr(logging=False, **kwargs)

    def __str__(self):
        return self.label()


# Class Facility
class Facility(BaseClass):
    def __init__(self, name=None, category=None, **kwargs):
        self.name = name
        self.category = category
        super().__init__(name=self.name, category=self.category)
        self.set_attr(logging=False, **kwargs)

    def __str__(self):
        return self.label()


# Class Instrument
class Instrument(BaseClass):
    def __init__(self, name=None, category=None, **kwargs):
        self.name = name
        self.category = category
        super().__init__(name=self.name, category=self.category)
        self.set_attr(logging=False, **kwargs)

    def __str__(self):
        return self.label()


# Class Experiment
class Experiment(BaseClass):
    def __init__(self, name=None, category=None, **kwargs):
        self.name = name
        self.category = category
        super().__init__(name=self.name, category=self.category)
        self.set_attr(logging=False, **kwargs)

    def __str__(self):
        return self.label()


# create the Dataset class
class Dataset(object):
    def __init__(self, **kwargs):
        self.data_path = kwargs.pop('data_path', None)
        self.dt_fr = kwargs.pop('dt_fr', None)
        self.dt_to = kwargs.pop('dt_to', None)
        self.database = kwargs.pop('database', Database(name='temporary', category='local'))
        self.facility = kwargs.pop('facility', Facility())
        self.instrument = kwargs.pop('instrument_opt', Instrument())
        self.experiment = kwargs.pop('experiment', Experiment())
        self.variables = None
        self._visual = kwargs.pop('visual', True)

    def add_variable(self, varname, **kwargs):
        opt_visual = kwargs.pop('opt_visual', {})
        self.variables[varname] = Variable(**kwargs)
        if self._visual:
            self.variables[varname].set_attr('visual', Visual(**opt_visual))

    def assign_data(self, Loader, opt_Loader=None):
        load_obj = Loader(opt_Loader)
        self.variables

    def pickup_variable(self, **kwargs):
        varname = kwargs.pop('varname', None)
        if varname in self.variables.keys():
            return self.variables[varname]
        else:
            raise KeyError(varname)

    def label(self):
        pass

    def __str__(self):
        return self.label()