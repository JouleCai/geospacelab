# import geospacelab.datahub._init_variable as BaseVariable
import copy
import importlib

import pathlib

import geospacelab.toolbox.utilities.pyclass as pyclass
import geospacelab.toolbox.utilities.pybasic as pybasic
from geospacelab.datahub._init_dataset import Dataset

datahub_root_dir = pathlib.Path(__file__).parent.absolute()

def example():

    var_config_list = []    # List of the variable configurations

    database_name = 'madrigal'
    facility_name = 'eiscat'
    assign_options = {'site': 'TRO', 'experiment': 'unknown', 'pulse_code': 'unknown', 'antenna':'UHF'}

    var_config = VariableConfig(source_keys=[database_name, facility_name], assign_options=assign_options)

    var_config_list.append(var_config.update(variable='n_e'))
    var_config_list.append(var_config.update(variable='T_i'))


class Datasets(object):
    def __init__(self):
        self._datasets ={}

        self.datasets = []
        self.hashes = []
        self.keys = []

    def __setitem__(self, key, value):
        if issubclass(value.__class__, Dataset):
            self._datasets[key] = value
        else:
            raise TypeError

    def __getitem__(self, key):
        return self._datasets[key]

    def get(self, **kwargs):
        assign_keys = kwargs.pop('assign_keys', [])
        module_keys = ['datahub']
        module_keys.extend(assign_keys)
        module = importlib.import_module('.'.join(module_keys) + '__init__.py')
        dataset = module.set_dataset(**kwargs.pop('options'))




        dataset_key = pybasic.dict_key_tree_plain()
        dataset_hash = pybasic.string_to_hash(dataset_key)
        try:
            ind_hash = self.hashes.index(dataset_hash)
            dataset = self.datasets[ind_hash]
        except ValueError:
            dataset = Dataset(**config_dict)
            self.datasets.append(dataset)
            self.hashes.append(dataset_hash)
            self.keys.append(dataset_key)


class DataHub(object):

    def __init__(self, dt_fr, dt_to, var_config_list=None):
        self.variables = Variables()
        self.datasets = Datasets()

        self.dt_fr = dt_fr
        self.dt_to = dt_to

        if var_config_list is not None:
            self.assign_variables(var_config_list=var_config_list)

    def assign_variables(self, var_config_list=None):

        for var_config in var_config_list:
            dataset = self.datasets.get(**var_config)

    def get_dataset(self, var_config):
        def generate_dataset_hash(var_config):

        dataset_key = generate_dataset_key(var_config)
        self.datasets



    def assign_variables(self, var_config_list):
        for var_config in var_config_list:
            dataset = self.get_dataset(var_config)




    def add_dataset(self, dataset=None, **kwargs):
        dataset_key = kwargs.pop('dataset_key', '')
        if dataset is None:
            # create a new dataset
            dataset = Dataset(**kwargs)
        elif not issubclass(dataset.__class__, Dataset):
            raise TypeError

        if not bool(dataset_key):
            dataset_key = dataset.label()

        self.__setitem__(dataset_key, dataset)


class VariableConfig(object):
    def __init__(self, source_keys=None, var_name='', assign_options=None):

        self.source_keys = source_keys
        self.var_name = var_name
        self.assign_options = assign_options

    def update(self, **kwargs):
        pyclass.set_object_attributes(self, **kwargs, append=False, logging=False)
        return copy.deepcopy(self)  # return a copied objective





if __name__ == "__main__":
    example()