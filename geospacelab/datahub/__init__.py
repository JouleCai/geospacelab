# import geospacelab.datahub._init_variable as BaseVariable
import copy
import importlib
import datetime

import pathlib

import geospacelab.toolbox.utilities.pyclass as pyclass
import geospacelab.toolbox.utilities.pybasic as pybasic
import geospacelab.toolbox.utilities.pylogging as mylog
from geospacelab.datahub.__init_metadata import *
from geospacelab.datahub.__init_dataset import DatasetModel
from geospacelab.datahub.__init_variable import VariableModel
import geospacelab.config.preferences as pfr


def example():
    pfr.datahub_data_root_dir = pathlib.Path('~/01-Work/00-Data')
    dt_fr = datetime.datetime.strptime('20210309' + '0000', '%Y%m%d%H%M')
    dt_to = datetime.datetime.strptime('20210309' + '2359', '%Y%m%d%H%M')
    database_name = 'madrigal'
    facility_name = 'eiscat'

    dh = DataHub(dt_fr, dt_to)
    ds_1 = dh.set_dataset(datasource_contents=['madrigal', 'eiscat'],
                          site='UHF', antenna='UHF', modulation='ant', data_file_type='eiscat-hdf5')
    var_1 = dh.assign_variable('n_e')
    var_2 = dh.assign_variable('T_i')


class DataHub(object):
    def __init__(self, dt_fr=None, dt_to=None, visual='off', **kwargs):
        self.dt_fr = dt_fr
        self.dt_to = dt_to
        self.visual = visual
        self.datasets = []
        self.variables = []

        super().__init__(**kwargs)

    def set_dataset(self, **kwargs):
        # set default starting and ending times
        kwargs.setdefault('dt_fr', self.dt_fr)
        kwargs.setdefault('dt_to', self.dt_to)
        kwargs.setdefault('visual', self.visual)

        datasource_mode = kwargs.pop('datasource_mode', 'sourced')  # ['sourced'], 'temporary', 'custom'
        datasource_contents = kwargs.pop('datasource_contents', [])
        dataset = kwargs.pop('dataset', None)

        append = True
        if datasource_mode == 'sourced':

            module_keys = [pfr.package_name, 'datahub', 'sources']
            module_keys.extend(datasource_contents)
            module = importlib.import_module('.'.join(module_keys))
            dataset = module.Dataset(**kwargs)

            if dataset.label() in [ds.label() for ds in self.datasets]:
                mylog.simpleinfo.info('The same dataset has been added!')
                append = False
            else:
                dataset.load_data()
        elif datasource_mode == 'temporary':
            if dataset is None:
                dataset = DatasetModel(database='temporary', **kwargs)
            elif issubclass(dataset, DatasetModel):
                pass
            else:
                raise TypeError

        if append:
            self.datasets.append(dataset)

        return dataset

    def assign_variable(self, var_name, **kwargs):
        dataset = kwargs.pop('dataset', None)
        if dataset is None:
            dataset = self.datasets[-1]  # the latest added dataset
        elif not issubclass(dataset.__class__, DatasetModel):
            raise TypeError

        var = dataset.set_variable(var_name)
        var.config(**kwargs)

        self.variables.append(var)

        return var

    def list_datasets(self):
        # print header
        mylog.simpleinfo.info("Listing datasets ...")
        mylog.simpleinfo.info('{:^20s}{:60s}'.format('Index', 'Dataset'))
        for ind, dataset in enumerate(self.datasets):
            dataset_label = dataset.label()
            mylog.simpleinfo.info('{:^20d}{:60s}'.format(ind, dataset_label))
        print()

    def list_assigned_variables(self):
        # print header
        mylog.simpleinfo.info("Listing the assigned variables ...")

        mylog.simpleinfo.info('{:^20s}{:30s}{:60s}'.format('Index', 'Variable name', 'Dataset'))
        for ind, var in enumerate(self.variables):
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