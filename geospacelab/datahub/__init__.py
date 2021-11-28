# Licensed under the BSD 3-Clause License
# Copyright (C) 2021 GeospaceLab (geospacelab)
# Author: Lei Cai, Space Physics and Astronomy, University of Oulu

__author__ = "Lei Cai"
__copyright__ = "Copyright 2021, GeospaceLab"
__license__ = "BSD-3-Clause License"
__email__ = "lei.cai@oulu.fi"
__docformat__ = "reStructureText"

import importlib
import datetime
import pathlib

import geospacelab.toolbox.utilities.pylogging as mylog
import geospacelab.toolbox.utilities.pybasic as pybasic
from geospacelab.datahub._base_metadata import *
from geospacelab.datahub._base_dataset import DatasetModel
from geospacelab.datahub._base_variable import VariableModel
from geospacelab import preferences as pfr


def example():
    dt_fr = datetime.datetime.strptime('20210309' + '0000', '%Y%m%d%H%M')
    dt_to = datetime.datetime.strptime('20210309' + '2359', '%Y%m%d%H%M')
    database_name = 'madrigal'
    facility_name = 'eiscat'

    dh = DataHub(dt_fr, dt_to)
    dh.list_sourced_datasets()
    ds_1 = dh.dock(datasource_contents=[database_name, facility_name],
                          site='UHF', antenna='UHF', modulation='ant', data_file_type='eiscat-hdf5')
    ds_1.load_data()
    var_1 = dh.assign_variable('n_e')
    var_2 = dh.assign_variable('T_i')


class DataHub(object):
    """The core of the data manager in GeospaceLab to dock different datasets.

    :param dt_fr: starting time.
    :type dt_fr: datetime.datetime
    :param dt_to: stopping time.
    :type dt_to: datetime.datetime
    :param visual: decide if a Visual object will be added to the variable objects. Options:['off'], 'on'.
    :type visual: str
    :param datasets: A dictionary that stores the datasets docked or added to the datahub. \
The keys (indices) are the integer numbers starting from 1.
    :type datasets: dict
    :param variables: A dictionary that stores the variables assigned from their own datasets to the datahub. \
Typically used for the viewers or outputs.
    :type variables: dict

    ========
    :example:

    """
    def __init__(self, dt_fr=None, dt_to=None, visual='off', **kwargs):
        """
        :param dt_fr: starting time, see above
        :param dt_to: stopping time, see above
        :param visual: visual attribute, see above
        :param kwargs: other keywords, used for class inheritance.
        """
        self.dt_fr = dt_fr
        self.dt_to = dt_to
        self.visual = visual
        self.datasets = {}
        self.variables = {}

        super().__init__(**kwargs)

    def dock(self, datasource_contents, **kwargs):
        """To dock a sourced dataset.

        :param datasource_contents: the contents that required for docking a sourced dataset. \
To look up the sourced dataset and the associated contents, call :func:`~geospacelab.datahub.DataHub.list_sourced_datasets()`.
        :type datasource_contents: list
        :param dt_fr: starting time, optional, use datahub.dt_fr if not specified.
        :type dt_fr: datetime.datetime
        :param dt_to: stopping time, optional, use datahub.dt_to if not specified.
        :type dt_to: datetime.datetime
        :param visual: variable attribute, use datahub.visual if not specified.
        :type visual: str
        :return: a dataset
        :rtype: object of :class:`~geospacelab.datahub.DatasetModel`
        """
        kwargs.setdefault('dt_fr', self.dt_fr)
        kwargs.setdefault('dt_to', self.dt_to)
        kwargs.setdefault('visual', self.visual)
        append = True

        module_keys = [pfr.package_name, 'datahub', 'sources']
        module_keys.extend(datasource_contents)
        try:
            module = importlib.import_module('.'.join(module_keys))
            dataset = getattr(module, 'Dataset')(**kwargs)
            dataset.kind = 'sourced'
        except ImportError or ModuleNotFoundError:
            mylog.StreamLogger.error(
                'The data source cannot be docked. \n'
                + 'Check the built-in sourced data using the method: "list_sourced_dataset". \n'
                + 'Or, add a temporary or user-defined dataset by the method of "add_dataset". '
            )
            return

        # if dataset.label() in [ds.label() for ds in self.datasets]:
        #     mylog.simpleinfo.info('The same dataset has been docked!')
        #    append = False

        if append:
            self._append_dataset(dataset)

        return dataset

    def add_dataset(self, *args, kind='temporary', dataset_class=DatasetModel, **kwargs):
        """Add one or multiple datasets, which is not sourced in the package.

        :param args: a list of the datasets.
        :param kind: the kind of a dataset, options: ['temporary'], or 'user-defined'. \
if temporary, a new dataset will be created from the DatasetModel.
        :param dataset_class: The dataset class as a model.
        :type dataset_class: DatasetModel subclass.
        :param kwargs: other keywords
        :return: None
        """
        kwargs.setdefault('dt_fr', self.dt_fr)
        kwargs.setdefault('dt_to', self.dt_to)
        kwargs.setdefault('visual', self.visual)
        kwargs.setdefault('datasets', [])

        if kind == 'temporary':
            kwargs.pop('datasets', None)
            dataset = dataset_class(name='temporary', **kwargs)
            self._append_dataset(dataset)
            return dataset

        for dataset in args:
            kwargs['datasets'].append(dataset)

        for dataset in kwargs['datasets']:
            kind = 'user-defined'
            self._append_dataset(dataset)
            if issubclass(dataset.__class__, DatasetModel):
                dataset.kind = kind
            else:
                TypeError('A dataset instance\'s class must be a heritage of the class DatasetModel!')

        return None

    def _append_dataset(self, dataset):
        """Append a dataset to the datahub

        :param dataset: a dataset (Dataset object)
        :return: None
        """
        ind = len(self.datasets.keys()) + 1
        name = 'dataset_{:02d}'.format(ind)
        if dataset.name is None:
            dataset.name = name
        self.datasets[ind] = dataset
        self._latest_dataset_ind = ind

    def get_variable(self, var_name, dataset=None, dataset_index=None):
        """To get a variable from the docked or added dataset.

        :param var_name: the name of the queried variable
        :param dataset: the dataset storing the queried variable.
        :param dataset_index: the index of the dataset in datahub.datasets. \
if both dataset or dataset_index are not specified, the function will get the \
variable from the assigned variables.
        :return: object of :class:`~geospacelab.datahub.VariableModel` or None if not existing.
        """
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
            mylog.StreamLogger.warning(
                f"Cannot find the queried variable in {repr(dataset)}. Return None.")
            var = None

        return var

    def assign_variable(self, var_name, dataset=None, dataset_index=None, add_new=False, **kwargs):
        """Assign a variable from the docked or added dataset.

        :param var_name: The name of the variable
        :param dataset: The dataset that stores the variable
        :param dataset_index: The index of the dataset in the datahub.datasets.
        :param add_new: if True, add the variable to the specified dataset and assign to the datahub
        :param kwargs: other keywords to configure the attributes of the variable.
        :return: object of :class:`~geospacelab.datahub.VariableModel`
        """
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
        """ append a variable to datahub.variables

        :param var: a object of :class:`~geospacelab.datahub.VariableModel`
        :return: None
        """
        ind = len(self.variables.keys()) + 1
        self.variables[ind] = var

    @staticmethod
    def list_sourced_datasets():
        """To list all the sourced datasets bult-in this package

        The list will be printed in the python console in a \"tree\" view.
        """
        this_file_dir = pathlib.Path(__file__).resolve().parent
        data_source_dir = this_file_dir / 'sources'
        sub_dirs = list(data_source_dir.glob("**"))
        data_sources = {}
        for sub_dir in sub_dirs:
            init_file = sub_dir / "__init__.py"
            if not init_file.is_file():
                continue
            module_name = str(sub_dir).replace('/', '.').split(pfr.package_name + '.', 1)[-1]
            try:
                module = importlib.import_module(module_name)
            except:
                continue
            try:
                getattr(module, 'Dataset')
            except AttributeError:
                continue

            contents = module_name.split('sources.')[-1].split('.')
            current_dict = data_sources
            try:
                required_inputs = getattr(module, 'default_attrs_required')
            except AttributeError:
                required_inputs = []

            for ind, content in enumerate(contents):
                content = content.upper()
                current_dict.setdefault(content, {})
                if ind == len(contents) - 1:
                    required_inputs = ['datasource_contents', *required_inputs]
                    current_dict[content] = {
                        'Required inputs when load_mode="AUTO"': required_inputs,
                        'datasource_contents': contents,
                    }
                    continue
                current_dict = current_dict[content]

        pybasic.dict_print_tree(data_sources, full_value=False, dict_repr=False, value_repr=True, max_level=None)

    def list_datasets(self):
        """ To list all the datasets that have been docked or added to the datahub

        The list will be printed in the console as a table
        """
        # print header
        mylog.simpleinfo.info("Listing datasets ...")
        mylog.simpleinfo.info('{:^20s}{:60s}'.format('Index', 'Dataset'))
        for ind, dataset in self.datasets.items():
            dataset_label = dataset.label()
            mylog.simpleinfo.info('{:^20d}{:60s}'.format(ind, dataset_label))
        print()

    def list_assigned_variables(self):
        """ To list all the assigned variables that have been docked or added to the datahub

        The list will be printed in the console as a table
        """
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

    def __repr__(self):
        rep = f"GeospaceLab DataHub object <starting time: {str(self.dt_fr)} and stopping time: {str(self.dt_to)}>"
        return rep


if __name__ == "__main__":
    example()