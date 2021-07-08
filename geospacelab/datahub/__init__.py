# import geospacelab.datahub._init_variable as BaseVariable
import copy

import geospacelab.toolbox.utilities.pyclass as pyclass
import geospacelab.toolbox.utilities.pybasic as pybasic
from geospacelab.datahub._init_dataset import Dataset


def example():

    vc_list = []    # List of the variable configurations

    database_config = {'name': 'madrigal'}  # configuration for the database, database's name if a string
    facility_config = {'name': 'eiscat'}    # configuration for the facility
    instrument_config = {'name': 'UHF'}     # configuration for the instrument
    experiment_config = {'name': 'cp2'}     # configuration for the experiment

    var_config = VariableConfig(database=database_config, facility=facility_config, instrument=instrument_config,
                                experiment=experiment_config)

    vc_list.extend([var_config.update(variable='n_e')])
    vc_list.extend([var_config.update(variable='T_i')])

    dh = DataHub()
    dh.add_variables(vc_list)

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

    def get(self, config_dict):
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
    datasets = Datasets()

    def __init__(self):
        self._variables = {}

    def __setitem__(self, key, value):
        if issubclass(value.__class__, Dataset):
            self._variables[key] = value
        else:
            raise TypeError

    def __getitem__(self, key):
        return self._variables[key]

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
    def __init__(self, database=None, facility=None, instrument=None, experiment=None, variable=None):
        if database is None:
            database = {'name': 'temporary'}

        self.database = self.input_validation(database)
        self.facility = self.input_validation(facility)
        self.instrument = self.input_validation(instrument)
        self.experiment = self.input_validation(experiment)
        self.variable = self.input_validation(variable)

    @staticmethod
    def input_validation(value_in):
        if isinstance(value_in, str):
            value_in = {'name', value_in}
        elif isinstance(value_in, dict):
            value_in = value_in
        else:
            return ValueError('Input must be either a string or a dictionary')
        return value_in

    def update(self, **kwargs):
        pyclass.set_object_attributes(self, **kwargs, append=False, logging=False)
        return copy.deepcopy(self)  # return a copied objective





if __name__ == "__main__":
    example()