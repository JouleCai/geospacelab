# import geospacelab.datahub._init_variable as BaseVariable
import copy
import importlib
import datetime

import pathlib

import geospacelab.toolbox.utilities.pyclass as pyclass
import geospacelab.toolbox.utilities.pybasic as pybasic
import geospacelab.toolbox.utilities.pylogging as mylog
from geospacelab.datahub.metadata_base import *
from geospacelab.datahub.dataset_base import DatasetModel
from geospacelab.datahub.variable_base import VariableModel
from geospacelab import preferences as pfr


def example():
    dt_fr = datetime.datetime.strptime('20210309' + '0000', '%Y%m%d%H%M')
    dt_to = datetime.datetime.strptime('20210309' + '2359', '%Y%m%d%H%M')
    database_name = 'madrigal'
    facility_name = 'eiscat'

    dh = DataHub(dt_fr, dt_to)
    ds_1 = dh.dock(datasource_contents=['madrigal', 'eiscat'],
                          site='UHF', antenna='UHF', modulation='ant', data_file_type='eiscat-hdf5')
    ds_1.load_data()
    var_1 = dh.assign_variable('n_e')
    var_2 = dh.assign_variable('T_i')


class DataHub(object):
    def __init__(self, dt_fr=None, dt_to=None, visual='off', **kwargs):
        self.dt_fr = dt_fr
        self.dt_to = dt_to
        self.visual = visual
        self.datasets = {}
        self.variables = {}

        super().__init__(**kwargs)

    def dock(self, datasource_contents, **kwargs):
        kwargs.setdefault('dt_fr', self.dt_fr)
        kwargs.setdefault('dt_to', self.dt_to)
        kwargs.setdefault('visual', self.visual)
        append = True

        module_keys = [pfr.package_name, 'datahub', 'sources']
        module_keys.extend(datasource_contents)
        try:
            module = importlib.import_module('.'.join(module_keys))
            dataset = module.Dataset(**kwargs)
            dataset.kind = 'sourced'
        except ImportError or ModuleNotFoundError:
            mylog.StreamLogger.error(
                'The data source cannot be docked. \n'
                + 'Check the built-in sourced data using the method: "list_sourced_dataset". \n'
                + 'Or, add a temporary or user-defined dataset by the method of "add_dataset". '
            )

        # if dataset.label() in [ds.label() for ds in self.datasets]:
        #     mylog.simpleinfo.info('The same dataset has been docked!')
        #    append = False

        if append:
            self._append_dataset(dataset)

        return dataset

    def add_dataset(self, *args, kind='temporary', **kwargs):
        kwargs.setdefault('dt_fr', self.dt_fr)
        kwargs.setdefault('dt_to', self.dt_to)
        kwargs.setdefault('visual', self.visual)
        kwargs.setdefault('datasets', [])

        for dataset in args:
            kwargs['datasets'].append(dataset)

        for dataset in kwargs['datasets']:
            kind = 'user-defined'
            self._append_dataset(dataset)
            if issubclass(dataset.__class__, DatasetModel):
                dataset.kind = kind
            else:
                TypeError('A dataset instance\'s class must be a heritage of the class DatasetModel!')

        if kind == 'temporary':
            dataset = DatasetModel(name='temporary', **kwargs)
            self._append_dataset(dataset)
            return dataset

    def _append_dataset(self, dataset):
        ind = len(self.datasets.keys()) + 1
        name = 'dataset_{:02d}'.format(ind)
        if dataset.name is None:
            dataset.name = name
        self.datasets[ind] = dataset
        self._latest_dataset_ind = ind

    def get_variable(self, var_name, dataset=None, dataset_index=None):
        if dataset is None and dataset_index is None:
            var = None
            for ind, var in self.variables.items():
                if var_name == var.name:
                    var = self.variables[ind]
            if var is None:
                mylog.StreamLogger.warning("Cannot find the variable in the datahub assigned variables. Try specify the dataset.")
            return var

        if dataset is None:
            dataset = self.datasets[dataset_index]
        elif not issubclass(dataset.__class__, DatasetModel):
            raise TypeError('A dataset instance\'s class must be a heritage of the class DatasetModel!')

        if dataset.exist(var_name):
            var = dataset[var_name]
        else:
            var = None

        return var

    def assign_variable(self, var_name, dataset=None, dataset_index=None, add_new=False, **kwargs):
        if dataset is None:
            if dataset_index is None:
                dataset = self.datasets[self._latest_dataset_ind]  # the latest added dataset
            else:
                dataset = self.datasets[dataset_index]
        elif not issubclass(dataset.__class__, DatasetModel):
            raise TypeError('A dataset instance\'s class must be a heritage of the class DatasetModel!')

        var = self.get_variable(var_name, dataset=dataset, dataset_index=dataset_index)

        if var is None:
            if add_new:
                var = dataset.add_variable(var_name)
            else:
                raise KeyError('The variable does not exist in the dataset. Set add_new=True, if you want to add.')
        var.config(**kwargs)

        if var not in self.variables.values():
            self._append_variable(var)

        return var

    def _append_variable(self, var):
        ind = len(self.variables.keys()) + 1
        self.variables[ind] = var

    def list_datasets(self):
        # print header
        mylog.simpleinfo.info("Listing datasets ...")
        mylog.simpleinfo.info('{:^20s}{:60s}'.format('Index', 'Dataset'))
        for ind, dataset in self.datasets.items():
            dataset_label = dataset.label()
            mylog.simpleinfo.info('{:^20d}{:60s}'.format(ind, dataset_label))
        print()

    def list_assigned_variables(self):
        # print header
        mylog.simpleinfo.info("Listing the assigned variables ...")

        mylog.simpleinfo.info('{:^20s}{:30s}{:60s}'.format('Index', 'Variable name', 'Dataset'))
        for ind, var in self.variables.items():
            dataset_label = var.dataset.label()
            var_name = var.name
            mylog.simpleinfo.info('{:^20d}{:30s}{:60s}'.format(ind, var_name, dataset_label))
        print()

    def save_to_pickle(self):
        pass

    def save_to_cdf(self):
        pass


# class DataHub(object):
#
#     def __init__(self, dt_fr, dt_to):
#         self.variables = {}
#         self.datasets = {}
#
#         self.dt_fr = dt_fr
#         self.dt_to = dt_to
#
#     def add_dataset(self, *args, config=None, **kwargs):
#         dataset = kwargs.pop('dataset', None)
#         if config is None:
#             if dataset is None:
#                 config = DatasetConfiguration(**kwargs)
#         else:
#             module_keys = [pfr.package_name, 'datahub', 'sources']
#             module_keys.extend(config.source_labels)
#             module = importlib.import_module('.'.join(module_keys))
#             dataset = module.Dataset(config=config, **kwargs)
#
#             if dataset in self.datasets.keys():
#                 mylog.simpleinfo.warning('The same dataset has been added!')
#             else:
#                 dataset.load()
#                 ind = len(self.datasets.keys()) + 1
#                 self.datasets[ind] = dataset
#
#         return dataset
#
#     def add_variable(self, config=None, **kwargs):
#         variable = kwargs.pop('variable', None)
#         dataset = kwargs.pop('dataset', None)
#         dataset_ind = kwargs.pop('dataset_ind', None)
#         if dataset is None:
#             # use the last added dataset
#             if dataset_ind is not None:
#                 dataset = self.datasets[dataset_ind]
#             else:
#                 dataset = self.datasets[len(self.datasets.keys())] # the latest added dataset
#         elif dataset not in self.datasets.values():
#                 self.add_dataset(dataset=dataset)
#
#         if config is not None:
#             if issubclass(config, VariableConfiguration):
#                 variable = dataset.set_variable(config=config)
#             else:
#                 raise TypeError
#         elif variable is not None:
#             if issubclass(variable, VariableBase):
#                 variable = variable
#             else:
#                 raise TypeError
#         return variable



#
# class DatasetConfiguration(object):
#     def __init__(self, **kwargs):
#         self.source_labels = kwargs.pop('source_labels', [])
#         self.set_attr(**kwargs)
#
#     def set_attr(self, **kwargs):
#         pyclass.set_object_attributes(self, **kwargs, append=True, logging=False)
#
#     def replace(self, **kwargs):
#         new_obj = DatasetConfiguration()
#         new_obj.__dict__.update(self.__dict__)
#         pyclass.set_object_attributes(new_obj, **kwargs, append=False, logging=False)
#         return new_obj
#
#
# class VariableConfiguration(object):
#     def __init__(self, **kwargs):
#         self.name = kwargs.pop('name', '')
#         self.set_attr(**kwargs)
#
#     def set_attr(self, **kwargs):
#         pyclass.set_object_attributes(self, **kwargs, append=True, logging=False)
#
#     def replace(self, **kwargs):
#         new_obj = VariableConfiguration()
#         new_obj.__dict__.update(self.__dict__)
#         pyclass.set_object_attributes(new_obj, **kwargs, append=False, logging=False)
#         return new_obj






if __name__ == "__main__":
    example()