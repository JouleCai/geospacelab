# import geospacelab.datahub._init_variable as BaseVariable
import copy
import importlib
import datetime

import pathlib

import geospacelab.toolbox.utilities.pyclass as pyclass
import geospacelab.toolbox.utilities.pybasic as pybasic
import geospacelab.toolbox.utilities.pylogging as mylog
from geospacelab.datahub._init_dataset import DatasetBase
import geospacelab.config.preferences as pfr

datahub_root_dir = pathlib.Path(__file__).parent.absolute()

def example():

    dt_fr = datetime.datetime.strptime('20210309' + '0000', '%Y%m%d%H%M')
    dt_to = datetime.datetime.strptime('20210309' + '2400', '%Y%m%d%H%M')
    database_name = 'madrigal'
    facility_name = 'eiscat'

    config_d = DatasetConfiguration(source_labels=[database_name, facility_name],
                                          site='TRO', antenna='UHF')
    config_v = VariableConfiguration()

    dh = DataHub(dt_fr, dt_to)
    dataset_label = dh.add_dataset(config_d)
    v1 = dh.add_variables(config_v.replace('n_e'))
    v2 = dh.add_variables(config_v.replace('T_i'))


class DataHub(object):

    def __init__(self, dt_fr, dt_to):
        self.variables = {}
        self.datasets = {}

        self.dt_fr = dt_fr
        self.dt_to = dt_to

    def add_dataset(self, *args, config=None, **kwargs):
        dataset = kwargs.pop('dataset', None)
        if config is None:
            if dataset is None:
                config = DatasetConfiguration(**kwargs)
            elif issubclass(dataset, DatasetBase):
                raise TypeError
        elif not isinstance(config, DatasetConfiguration):
            raise TypeError

        if config is not None:
            module_keys = [pfr.package_name, 'datahub', 'sources']
            module_keys.extend(config.source_labels)
            module = importlib.import_module('.'.join(module_keys))
            dataset = module.Dataset(config=config, **kwargs)

            if dataset in self.datasets.keys():
                mylog.simpleinfo.warning('The same dataset has been added!')
            else:
                dataset.load()
                ind = len(self.datasets.keys()) + 1
                self.datasets[ind] = dataset

        return dataset

    def add_variable(self, config=None, **kwargs):
        variable = kwargs.pop('variable', None)
        dataset = kwargs.pop('dataset', None)
        dataset_ind = kwargs.pop('dataset_ind', None)
        if dataset is None:
            # use the last added dataset
            if dataset_ind is not None:
                dataset = self.datasets[dataset_ind]
            else:
                dataset = self.datasets[len(self.datasets.keys())] # the latest added dataset
        elif issubclass(dataset, DatasetBase):
            if dataset not in self.datasets.values():
                self.add_dataset(dataset=dataset)
        else:
            raise TypeError

        if config is not None:
            if issubclass(config, VariableConfiguration):
                variable = dataset.set_variable(config=config)
            else:
                raise TypeError
        elif variable is not None:
            if issubclass(variable, VariableBase):
                variable = variable
            else:
                raise TypeError
        return variable


class DatasetConfiguration(object):
    def __init__(self, **kwargs):
        self.source_labels = kwargs.pop('source_labels', [])
        self.set_attr(**kwargs)

    def set_attr(self, **kwargs):
        pyclass.set_object_attributes(self, **kwargs, append=True, logging=False)

    def replace(self, **kwargs):
        new_obj = DatasetConfiguration()
        new_obj.__dict__.update(self.__dict__)
        pyclass.set_object_attributes(new_obj, **kwargs, append=False, logging=False)
        return new_obj


class VariableConfiguration(object):
    def __init__(self, **kwargs):
        self.name = kwargs.pop('name', '')
        self.set_attr(**kwargs)

    def set_attr(self, **kwargs):
        pyclass.set_object_attributes(self, **kwargs, append=True, logging=False)

    def replace(self, **kwargs):
        new_obj = VariableConfiguration()
        new_obj.__dict__.update(self.__dict__)
        pyclass.set_object_attributes(new_obj, **kwargs, append=False, logging=False)
        return new_obj






if __name__ == "__main__":
    example()