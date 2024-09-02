"""

The module datahub is the data manager in GeospaceLab. The datahub has three core class-based components:

    - :class:`DataHub <geospacelab.datahub.DataHub>` manages a set of datasets docked or added to the datahub.
    - :class:`Dataset <geospacelab.datahub.DatasetModel>` manages a set of variables loaded from a data source.
    - :class:`Variable <geospacelab.datahub.VariableModel>` records the value, error, and various attributes \
    (e.g., name, label, unit, depends, ndim, ...) of a variable.

"""

# Licensed under the BSD 3-Clause License
# Copyright (C) 2021 GeospaceLAB (geospacelab)
# Author: Lei Cai, Space Physics and Astronomy, University of Oulu

__author__ = "Lei Cai"
__copyright__ = "Copyright 2021, GeospaceLab"
__license__ = "BSD-3-Clause License"
__email__ = "lei.cai@oulu.fi"
__docformat__ = "reStructureText"

# __all__ = ['DataHub', 'DatasetModel', 'VariableModel',]

import importlib
import datetime
import pathlib

import geospacelab.toolbox.utilities.pylogging as mylog
import geospacelab.toolbox.utilities.pybasic as pybasic
from geospacelab.datahub.__metadata_base__ \
    import DatabaseModel, MetadataModel, FacilityModel, SiteModel, ProductModel, InstrumentModel
from geospacelab.datahub.__dataset_base__ import DatasetBase, DatasetUser, DatasetSourced
from geospacelab.datahub.__variable_base__ import Visual
from geospacelab.datahub.__variable_base__ import VariableBase as VariableModel
from geospacelab.config import pref as pfr


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


def create_datahub(dt_fr, dt_to, visual='off', datahub_class=None, **kwargs):
    """
    Create a datahub object.

    :param dt_fr: The starting time.
    :type dt_fr: datetime.datetime
    :param dt_to: The stopping time.
    :type dt_to: datetime.datetime
    :param visual: If "on", a Visual object is aggregated to the Variable object.
    :type visual: {'off', 'on'}, default: 'off'
    :param datahub_class: If ``None``, create a datahub object based on the
        default :class:`DataHub <geospacelab.datahub.DataHub>` class.
    :type datahub_class: DataHub or its subclass
    :param kwargs: Other optional keyword arguments as inputs to DataHub.
    :return: dh
    :rtype: DataHub object

    :Example:
    >>> import geospacelab.datahub as datahub
    >>> import datetime
    >>> dt_fr = datetime.datetime.strptime('20210309' + '0000', '%Y%m%d%H%M')
    >>> dt_to = datetime.datetime.strptime('20210309' + '2359', '%Y%m%d%H%M')
    >>> dh = datahub.create_datahub(dt_fr, dt_to)

    :seealso:: :class:`DataHub <geospacelab.datahub.DataHub>`
    """

    if datahub_class is None:
        datahub_class = DataHub

    dh = datahub_class(dt_fr=dt_fr, dt_to=dt_to, visual=visual, **kwargs)

    return dh


class DataHub(object):
    """
    The class DataHub manage a set of datasets from various data sources.

    :ivar datetime.datetime dt_fr: The starting time.
    :ivar datetime.datetime dt_to: The ending time.
    :ivar str, {'off', 'on'} visual: If "on", a Visual object will be aggregated to the Variable object.
    :ivar dict datasets:  A *dict* stores the datasets added (:meth:`add_dataset`)
        or docked (:meth:`dock`) to the datahub.
    :ivar dict variables: A *dict* stores the variables assigned from their aggregated datasets.
        Typically used for the dashboards or the I/O configuration.

    **Usage**:

        - Create a DataHub object:

        :Example: Import the datahub module and create a DataHub object

            >>> import geospacelab.datahub as datahub
            >>> import datetime
            >>> dt_fr = datetime.datetime.strptime('20210309' + '0000', '%Y%m%d%H%M')
            >>> dt_to = datetime.datetime.strptime('20210309' + '2359', '%Y%m%d%H%M')
            >>> dh = datahub.DataHub(dt_fr, dt_to)

        :seealso:: :func:`create_datahub <geospacelab.datahub.create_datahub>`

        - Dock a built-in dataset:

        :Example: Dock a EISCAT dataset

            >>> database_name = 'madrigal'      # built-in sourced database name
            >>> facility_name = 'eiscat'
            >>> site = 'UHF'      # facility attributes required, check from the eiscat schedule page
            >>> antenna = 'UHF'
            >>> modulation = 'ant'
            >>> ds_1 = dh.dock(datasource_contents=[database_name, facility_name], site=site, antenna=antenna, modulation=modulation, data_file_type='eiscat-hdf5')

        :seealso:: :meth:`dock`

        - How to know ``datasource_contents`` and required inputs?

        :Example: List the buit-in data sources

            >>> dh.list_sourced_datasets()

        :seealso:: :meth:`list_sourced_datasets`

    """

    __dataset_model__ = DatasetBase
    __variable_model__ = VariableModel

    def __init__(self, dt_fr=None, dt_to=None, visual='off', **kwargs):
        """
            :param dt_fr: The starting time.
            :type dt_fr: datetime.datetime
            :param dt_to: The stopping time.
            :type dt_to: datetime.datetime
            :param visual: If "on", a Visual object is aggregated to the Variable object.
            :type visual: {'off', 'on'}, default: 'off'
            :param kwargs: other keyword arguments forwarded to the inherited class.
        """

        self.dt_fr = dt_fr
        self.dt_to = dt_to
        self.visual = visual
        self.datasets = {}
        self.variables = {}
        self._current_dataset = None
        self.done = False

        super().__init__(**kwargs)

    def dock(self, datasource_contents, **kwargs) -> DatasetSourced:
        """Dock a built-in or registered dataset.

        :param datasource_contents: the contents that required for docking a sourced dataset.
            To look up the sourced dataset and the associated contents, call :func:`~geospacelab.datahub.DataHub.list_sourced_datasets()`.
        :type datasource_contents: list
        :param dt_fr: starting time, optional, use datahub.dt_fr if not specified.
        :type dt_fr: datetime.datetime
        :param dt_to: stopping time, optional, use datahub.dt_to if not specified.
        :type dt_to: datetime.datetime
        :param visual: variable attribute, use datahub.visual if not specified.
        :type visual: str
        :return: ``dataset``
        :rtype: :class:`Dataset <geospacelab.datahub.DatasetModel>` object

        :seealso:: :meth:`add_dataset`

        :note:: The difference between the methods :meth:`dock` and :meth:`add_dataset` is that :meth:`dock` adds a built-in data
            source, while :meth:`add_dataset` add a temporary or user-defined dataset, which is not initially included in the
            package.

        -------------------

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
        except Exception as error:
            error_str = str(error)
            print(error_str.replace(pfr.package_name + '.datahub.sources.', ''))
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

    def add_dataset(self, *args, kind='temporary', dataset_class=None, **kwargs) -> __dataset_model__:
        """Add one or more datasets, which can be a "temporary" or "user-defined" dataset.

        :param args: A list of the datasets.
        :type args: list(dataset)
        :param kind: The type of a dataset. If temporary, a new dataset will be created from the DatasetModel.
        :type kind: {'temporary', 'user-defined'}, default: 'temporary'
        :param dataset_class: If None, the default class is DatasetModel. Used when ``kind='temporary'``.
        :type dataset_class: DatasetModel or its subclass
        :param kwargs: Other keyword arguments forwarded to ``dataset_class``
        :return: None

        :seealso:: :meth:`dock`

        """
        kwargs.setdefault('dt_fr', self.dt_fr)
        kwargs.setdefault('dt_to', self.dt_to)
        kwargs.setdefault('visual', self.visual)
        kwargs.setdefault('datasets', [])

        if dataset_class is None:
            dataset_class = self.__dataset_model__

        if kind == 'temporary':
            kwargs.pop('datasets', None)
            dataset = dataset_class(name='temporary', **kwargs)
            self._append_dataset(dataset)
            return dataset
        elif len(args) == 0 and kind in ["user-defined", "UserDefined", "UD"]:
            kwargs.pop('datasets', None)
            dataset = DatasetUser(name='', **kwargs)
            self._append_dataset(dataset)
            return dataset
        else:
            kind = "user-defined"

        for dataset in args:
            kwargs['datasets'].append(dataset)

        for dataset in kwargs['datasets']:
            kind = 'user-defined'
            if issubclass(dataset.__class__, DatasetUser):
                dataset.kind = kind
            else:
                TypeError('A dataset instance\'s class must be a heritage of DatasetUser!')
            self._append_dataset(dataset)

        return None

    def _append_dataset(self, dataset):
        """Append a dataset to the datahub

        :param dataset: a dataset (Dataset object)
        :return: None
        """

        ind = len(self.datasets.keys())
        name = 'dataset_{:02d}'.format(ind)
        if dataset.name is None:
            dataset.name = name
        self.datasets[ind] = dataset
        self.set_current_dataset(dataset=dataset)

    def set_current_dataset(self, dataset=None, dataset_index=None):
        """
        Set the current dataset.

        :param dataset: A Dataset object.
        :param dataset_index: The index of the dataset in ``.datasets``.
        :type dataset_index: int
        :rtype: None
        """
        if dataset is not None:
            if dataset in self.datasets.values():
                cds = dataset
            else:
                raise ValueError("The input dataset is not aggregated to the datahub!")
        elif dataset_index is not None:
            cds = self.datasets[dataset_index]
        else:
            raise ValueError("Either dataset or dataset_index must be set!")
        self._current_dataset = cds

    def get_current_dataset(self, index=False):
        """
        Get the current dataset.

        :param index: The index of the dataset.
        :type index: bool
        :return: If ``index=False``, dataset object, else dataset_index.
        """

        if index:
            datasets_r = {value: key for key, value in self.datasets.items()}
            res = datasets_r[self._current_dataset]
        else:
            res = self._current_dataset

        return res

    def get_variable(self, var_name, dataset=None, dataset_index=None) -> __variable_model__:
        """To get a variable from the docked or added dataset.

        :param str var_name: the name of the queried variable
        :param DatasetBase object dataset: the dataset storing the queried variable.
        :param int dataset_index: the index of the dataset in datahub.datasets.
            if both dataset or dataset_index are not specified, the function will get the
            variable from the current dataset.
        :return: var
        :rtype: :class:`VariableModel <geospacelab.datahub.VariableModel>` object or None

        :seealso: :meth:`assign_variable`

        :note:: Both :meth:`get_variable` and :meth:`assign_variable` return a variable object assigned from a dataset.
            The former only returns the object, and the latter also assign the variable to the ``DataHub.variables``.
        """

        if dataset is None and dataset_index is None:
            dataset = self.get_current_dataset()

        if dataset is None:
            dataset = self.datasets[dataset_index]
        elif not issubclass(dataset.__class__, DatasetBase):
            raise TypeError('A dataset instance\'s class must be an inheritance of DatasetBase!')

        if dataset.exist(var_name):
            var = dataset[var_name]
        else:
            mylog.StreamLogger.warning(
                f"Cannot find the queried variable in {repr(dataset)}. Return None.")
            var = None

        return var

    def assign_variable(self, var_name, dataset=None, dataset_index=None, add_new=False, **kwargs) -> __variable_model__:
        """Assign a variable to `DataHub.variables` from the docked or added dataset.

        :param var_name: The name of the variable
        :param dataset: The dataset that stores the variable
        :param dataset_index: The index of the dataset in the datahub.datasets.
        :param add_new: if True, add the variable to the specified dataset and assign to the datahub
        :param kwargs: other keywords to configure the attributes of the variable.
        :return: object of :class:`~geospacelab.datahub.VariableModel`

        :seealso:: :meth:`get_variable`

        """
        if dataset is None:
            if dataset_index is None:
                dataset = self.get_current_dataset()  # the current dataset
            else:
                dataset = self.datasets[dataset_index]
        elif not issubclass(dataset.__class__, DatasetBase):
            raise TypeError('A dataset instance\'s class must be a heritage of the class DatasetBase!')

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
        ind = len(self.variables.keys())
        self.variables[ind] = var

    @property
    def host_dataset(self):
        return self.datasets[0]

    @staticmethod
    def list_sourced_datasets():
        """List all the bult-in data sources this package

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
        """ List all the datasets that have been docked or added to the datahub

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
        """ List all the assigned variables that have been docked or added to the datahub

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
        name = self.__class__.__name__
        r =  f"GeospaceLab {name} object <starting time: {str(self.dt_fr)} and stopping time: {str(self.dt_to)}>"
        return r


if __name__ == "__main__":
    example()
