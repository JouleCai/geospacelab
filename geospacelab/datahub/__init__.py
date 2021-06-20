# import geospacelab.datahub._init_variable as BaseVariable

from geospacelab.datahub._init_dataset import Dataset


class DataHub(object):

    def __init__(self):
        self._datasets = {}

    def __setitem__(self, key, value):
        if issubclass(value.__class__, Dataset):
            self._datasets[key] = value
        else:
            raise TypeError

    def __getitem__(self, key):
        return self._datasets[key]

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


