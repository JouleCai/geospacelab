"""

The module datahub is the data manager in GeospaceLab. The datahub has three core class-based components:

    - :class:`DataHub <geospacelab.datahub.DataHub>` manages a set of datasets docked or added to the datahub.
    - :class:`Dataset <geospacelab.datahub.DatasetModel>` manages a set of variables loaded from a data source.
    - :class:`Variable <geospacelab.datahub.VariableModel>` records the value, error, and various attributes \
    (e.g., name, label, unit, depends, ndim, ...) of a variable.

"""

# Licensed under the BSD 3-Clause License
# Copyright (C) 2021 GeospaceLab (geospacelab)
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
from geospacelab.datahub._base_metadata import *
from geospacelab.datahub._base_dataset import DatasetModel
from geospacelab.datahub._base_variable import VariableModel
from geospacelab import preferences as pfr



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

    from geospacelab.datahub import DatasetBase, VariableBase
    __dataset_model__ = DatasetBase
    __variable_model__ = VarialeBase

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

        super().__init__(**kwargs)

    def dock(self, datasource_contents, **kwargs) -> __dataset_model__:
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
        except ImportError or ModuleNotFoundError as error:
            print(error)
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

        for dataset in args:
            kwargs['datasets'].append(dataset)

        for dataset in kwargs['datasets']:
            kind = 'user-defined'
            if issubclass(dataset.__class__, DatasetModel):
                dataset.kind = kind
            else:
                TypeError('A dataset instance\'s class must be a heritage of the class DatasetModel!')
            self._append_dataset(dataset)

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
        elif not issubclass(dataset.__class__, DatasetModel):
            raise TypeError('A dataset instance\'s class must be an inheritance of the class DatasetModel!')

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
        rep = f"GeospaceLab DataHub object <starting time: {str(self.dt_fr)} and stopping time: {str(self.dt_to)}>"
        return rep


class DatasetBase(object):
    """
    A dataset is a dictionary-like object used for downloading and loading data from a data source. The items in
    the dataset are the variables loaded from the data files. The parameters listed below are the general attributes used
    for the dataset class and its inheritances.

    :ivar str name: The name of the dataset.
    :ivar str kind: The type of the dataset. 'sourced': the data source has been added in the package,
      'temporary': a dataset added temporarily, 'user-defined': a data source defined by the user.
    :ivar datetime.datetime or None dt_fr: the starting time of the data records.
    :ivar datetime.datetime or None dt_fr: the starting time of the data records.
    :ivar str, {"on", "off"} visual: If "on", append the Visual object to the Variable object.
    :ivar list label_fields: A list of strings, indicating the fields used for generating the dataset label.
    """

    from geospacelab.datahub import VariableBase
    __variable_model__ = VariableBase

    def __init__(self,
                 dt_fr: datetime.datetime = None,
                 dt_to: datetime.datetime = None,
                 name: str = '',
                 kind: str = '',
                 visual: str = 'off',
                 label_fields: list = ('name', 'kind'), **kwargs):
        """
        Initial inputs to create the object.

            :param name: The name of the dataset.
            :type name: str or None
            :param kind: The type of the dataset. 'sourced': the data source has been added in the package,
              'temporary': a dataset added temporarily, 'user-defined': a data source defined by the user.
            :type kind: {'sourced', 'temporary', 'user-defined'}
            :param dt_fr: the starting time of the data records.
            :type dt_fr: datetime.datetime, default: None
            :param dt_to: the stopping time of the the data records.
            :type dt_to: datetime.datetime, default: None
            :param visual: If "on", append the attribute visual to the vairables.
            :type visual: {'on', 'off'}
            :param label_fields: A list of strings, indicating the fields used for generating the dataset label.
            :type label_fields: list, {['name', 'kind']}
        """

        self._variables = {}
        self.name = name
        self.kind = kind
        self.dt_fr = dt_fr
        self.dt_to = dt_to

        self.visual = visual

        self.label_fields = label_fields

    def add_variable(self, var_name: str, configured_variables=None, variable_class=None, **kwargs) -> __variable_model__:
        """
        Add a variable to the dataset.

        :param var_name:
        :param configured_variables:
        :param variable_class:
        :param kwargs:
        :return:
        """
        if variable_class is None:
            variable_class = self.__variable_model__
        if configured_variables is None:
            configured_variables = {}
        if var_name in configured_variables.keys():
            configured_variable = configured_variables[var_name]
            if type(configured_variable) is dict:
                self[var_name] = variable_class(**configured_variable)
            elif issubclass(configured_variable.__class__, VariableModel):
                self[var_name] = configured_variable.clone()
            self[var_name].dataset = self
            self[var_name].visual = self.visual
        else:
            self[var_name] = variable_class(dataset=self, visual=self.visual, **kwargs)
        return self[var_name]

    def label(self, fields=None, separator=' | ', lowercase=True) -> str:
        """
        Return a label of the data set.
        :param fields: The attribute names for the label.
        :param separator: A separator between two attributes.
        :param lowercase: Show lowercase letters only.
        :return: label
        """
        if fields is None:
            fields = self.label_fields
        sublabels = []
        for attr_name in fields:
            if not str(attr_name):
                sublabels.append('*')
            else:
                sublabels.append(getattr(self, attr_name))
        label = pybasic.str_join(*sublabels, separator=separator, lowercase=lowercase)
        return label

    def config(self, logging: bool = True, **kwargs) -> None:
        """
        Configure the attributes of the dataset.

        :param logging: Show logging if True.
        :param kwargs:
        :return:
        """
        pyclass.set_object_attributes(self, append=False, logging=logging, **kwargs)

    def add_attr(self, logging=True, **kwargs) -> None:
        pyclass.set_object_attributes(self, append=True, logging=logging, **kwargs)

    def attrs_to_dict(self, attr_names=None) -> dict:
        if attr_names is None:
            attr_names = []
        items = {}
        for attr_name in attr_names:
            items[attr_name] = getattr(self, attr_name)
        return items

    def list_all_variables(self) -> None:
        # print header
        label = self.label()
        mylog.simpleinfo.info("Dataset: {}".format(label))
        mylog.simpleinfo.info("Printing all of the variables ...")
        mylog.simpleinfo.info('{:^20s}{:^30s}'.format('No.', 'Variable name'))
        for ind, var_name in enumerate(self._variables.keys()):
            mylog.simpleinfo.info('{:^20d}{:30s}'.format(ind+1, var_name))
        mylog.simpleinfo.inf('')

    def keys(self) -> list:
        return list(self._variables.keys())

    def items(self) -> dict:
        return dict(self._variables.items())

    def __setitem__(self, key, value):
        if value is not None:
            if not issubclass(value.__class__, VariableModel):
                raise TypeError
        self._variables[key] = value

    def __getitem__(self, key) -> __variable_model__:
        # if key not in self._variables.keys():
        #    raise KeyError
        return self._variables[key]

    def __delitem__(self, key):
        del self._variables[key]
        pass

    def __repr__(self):
        rep = f"GeospaceLab Dataset object <{self.label()}, " + \
              f"starting time: {str(self.dt_fr)}, stopping time: {str(self.dt_to)}>"
        return rep.replace('geospacelab.datahub.sources.', '')

    # def add_variable(self, variable, name=None):
    #     if issubclass(variable, VariableModel):
    #         pass
    #     else:
    #         variable = VariableModel(value=variable, name=name, visual=self.visual)
    #     self[variable.name] = variable
    def exist(self, var_name) -> bool:
        if var_name in self._variables.keys():
            return True
        else:
            return False

    def remove_variable(self, name):
        del self[name]

    def get_variable_names(self) -> list:
        return list(self._variables.keys())

    def _set_default_variables(self, default_variable_names, configured_variables=None):
        if configured_variables is None:
            configured_variables = {}
        for var_name in default_variable_names:
            self.add_variable(var_name, configured_variables=configured_variables)


class DatasetUser(DatasetBase):

    def __init__(self,
                 dt_fr: datetime.datetime = None,
                 dt_to: datetime.datetime = None,
                 name: str = '',
                 visual: str = 'off',
                 label_fields: list = ('name', 'kind'), **kwargs):

        super().__init__(
            dt_fr=dt_fr, dt_to=dt_to, name=name, kind='user-defined', visual=visual, label_fields=label_fields, **kwargs
        )


class DatasetSourced(DatasetBase):

    def __init__(self,
                 dt_fr: datetime.datetime = None,
                 dt_to: datetime.datetime = None,
                 name: str = '',
                 visual: str = 'off',
                 label_fields: list = ('name', 'kind'), **kwargs):

        super().__init__(
            dt_fr=dt_fr, dt_to=dt_to, name=name, kind='sourced', visual=visual, label_fields=label_fields, **kwargs
        )

        self._variables = {}
        self.name = kwargs.pop('name', None)
        self.kind = kwargs.pop('kind', None)
        self.dt_fr = kwargs.pop('dt_fr', None)
        self.dt_to = kwargs.pop('dt_to', None)

        self.loader = kwargs.pop('loader', None)
        self.downloader = kwargs.pop('downloader', None)
        self.load_mode = kwargs.pop('load_mode', 'AUTO')  # ['AUTO'], 'dialog', 'assigned'
        self.data_root_dir = kwargs.pop('data_root_dir', pref.datahub_data_root_dir)
        self.data_file_paths = kwargs.pop('data_file_paths', [])
        self.data_file_num = kwargs.pop('data_file_num', 0)
        self.data_file_ext = kwargs.pop('data_file_ext', '*')
        self.data_search_recursive = kwargs.pop('data_search_recursive', False)
        self.visual = kwargs.pop('visual', 'off')
        self.time_clip = kwargs.pop('time_clip', True)

        self.label_fields = kwargs.pop('label_fields', [])

    def search_data_files(self, **kwargs):
        """
        Search the data files by the input pattern in the file name. The search method is based on pathlib.glob.
        For a dataset inheritance, a wrapper can be added for a custom setting.

        :param initial_file_dir: The initial file directory for searching.
        :type initial_file_dir: str or pathlib.Path, default: DatasetModel.data_root_dir.
        :param search_pattern: Unix style pathname pattern, see also pathlib.glob.
        :type search_pattern: str.
        :param recursive: Search recursively if True.
        :type recursive: bool, default: DatasetModel.data_search_recursive.
        :param allow_multiple_files: Allow multiple files as a result.
        :type allow_mulitple_files: bool, default: False.

        :return: done, bool, if False, no matches.
        """

        done = False
        initial_file_dir = kwargs.pop('initial_file_dir', self.data_root_dir)
        search_pattern = kwargs.pop('search_pattern', '*')
        recursive = kwargs.pop('recursive', self.data_search_recursive)
        allow_multiple_files = kwargs.pop('allow_multiple_files', False)
        if str(self.data_file_ext):
            search_pattern = search_pattern + '.' + self.data_file_ext
        if recursive:
            search_pattern = '**/' + search_pattern
        paths = list(initial_file_dir.glob(search_pattern))

        import natsort
        paths = natsort.natsorted(paths, reverse=False)
        if len(paths) == 1:
            done = True
            self.data_file_paths.append(paths[0])
        elif len(paths) > 1:
            if allow_multiple_files:
                done = True
                self.data_file_paths.extend(paths)
            else:
                mylog.StreamLogger.error("Multiple files found! Restrict the search condition.")
                print(paths)
                raise FileExistsError
        else:
            print('Cannot find the requested data file in {}'.format(initial_file_dir))

        return done

    def open_dialog(self, **kwargs):
        """
        Open a dialog to select the data files.
        """

        def tk_open_file():

            import tkinter as tk
            from tkinter import filedialog

            root = tk.Tk()
            root.withdraw()

            dialog_title = "Select a data file ..."

            if str(self.data_file_ext):
                p1 = '*.' + self.data_file_ext
            file_types = (('data files', p1), ('all files', '*.*'))
            file_name = filedialog.askopenfilename(
                title=dialog_title,
                initialdir=initial_file_dir,
                filetypes=file_types,
            )

            return file_name

        initial_file_dir = kwargs.pop('initial_file_dir', self.data_root_dir)

        if initial_file_dir is None:
            initial_file_dir = self.data_root_dir

        self.data_file_num = kwargs.pop('data_file_num', self.data_file_num)
        if self.data_file_num == 0:
            value = input("How many files will be loaded? Input the number: ")
            self.data_file_num = int(value)
        for nf in range(self.data_file_num):
            # value = input("Input the No. {} file's full path: ".format(str(nf)))
            value = tk_open_file()
            fp = pathlib.Path(value)
            if not fp.is_file():
                mylog.StreamLogger.warning("The input file does not exist!")
                return
            self.data_file_paths.append(fp)

    def check_data_files(self, **kwargs):
        """
        Check the existing of the data files before loading the data, depending on the loading mode (``load_mode``).
        This methods still needs to be improved as different datasets may have different variables as epochs. Two kinds
        of things can be done: 1. write a wrapper in the new dataset inheritance. 2. Add a script to recognize the
        epoch variables.

        """
        self.load_mode = kwargs.pop('load_mode', self.load_mode)
        if self.load_mode == 'AUTO':
            self.search_data_files(**kwargs)
        elif self.load_mode == 'dialog':
            self.open_dialog(**kwargs)
        elif self.load_mode == 'assigned':
            self.data_file_paths = kwargs.pop('data_file_paths', self.data_file_paths)
            self.data_file_paths = [pathlib.Path(f) for f in self.data_file_paths]
        else:
            raise NotImplementedError
        self.data_file_num = len(self.data_file_paths)

    def time_filter_by_range(self, var_datetime=None, var_datetime_name=None):
        """
        Clip the times.

        :param var_datetime:
        :param var_datetime_name:
        :return:
        """
        if var_datetime is None and var_datetime_name is None:
            var_datetime = self['DATETIME']
        if var_datetime_name is not None:
            var_datetime = self[var_datetime_name]
        if var_datetime.value is None:
            return
        inds = np.where((var_datetime.value.flatten() >= self.dt_fr) & (var_datetime.value.flatten() <= self.dt_to))[0]
        self.time_filter_by_inds(inds, var_datetime=var_datetime)

    def time_filter_by_inds(self, inds, var_datetime=None):
        if inds is None:
            return
        if not list(inds):
            return
        if var_datetime is None:
            var_datetime = self['DATETIME']

        shape_0 = var_datetime.value.shape[0]
        for var in self._variables.values():
            if var.value is None:
                continue
            if var.value.shape[0] == shape_0 and len(var.value.shape) > 1:
                var.value = var.value[inds, ::]

    def add_variable(self, var_name, configured_variables=None, variable_class=None, **kwargs):
        if variable_class is None:
            variable_class = self.__variable_model__
        if configured_variables is None:
            configured_variables = {}
        if var_name in configured_variables.keys():
            configured_variable = configured_variables[var_name]
            if type(configured_variable) is dict:
                self[var_name] = variable_class(**configured_variable)
            elif issubclass(configured_variable.__class__, VariableModel):
                self[var_name] = configured_variable.clone()
            self[var_name].dataset = self
            self[var_name].visual = self.visual
        else:
            self[var_name] = variable_class(dataset=self, visual=self.visual, **kwargs)
        return self[var_name]

    def label(self, fields=None, separator=' | ', lowercase=True):
        """
        Return a label of the data set.
        :param fields: The attribute names for the label.
        :param separator: A separator between two attributes.
        :param lowercase: Show lowercase letters only.
        :return: label
        """
        if fields is None:
            fields = self.label_fields
        sublabels = []
        for attr_name in fields:
            if not str(attr_name):
                sublabels.append('*')
            else:
                sublabels.append(getattr(self, attr_name))
        label = pybasic.str_join(*sublabels, separator=separator, lowercase=lowercase)
        return label

    def config(self, logging=True, **kwargs):
        """
        Configure the attributes of the dataset.

        :param logging: Show logging if True.
        :param kwargs:
        :return:
        """
        pyclass.set_object_attributes(self, append=False, logging=logging, **kwargs)

    def add_attr(self, logging=True, **kwargs):
        pyclass.set_object_attributes(self, append=True, logging=logging, **kwargs)

    def attrs_to_dict(self, attr_names=None):
        if attr_names is None:
            attr_names = []
        items = {}
        for attr_name in attr_names:
            items[attr_name] = getattr(self, attr_name)
        return items

    def list_all_variables(self):
        # print header
        label = self.label()
        mylog.simpleinfo.info("Dataset: {}".format(label))
        mylog.simpleinfo.info("Printing all of the variables ...")
        mylog.simpleinfo.info('{:^20s}{:^30s}'.format('No.', 'Variable name'))
        for ind, var_name in enumerate(self._variables.keys()):
            mylog.simpleinfo.info('{:^20d}{:30s}'.format(ind + 1, var_name))
        mylog.simpleinfo.inf('')

    def keys(self):
        return self._variables.keys()

    def items(self):
        return self._variables.items()

    def __setitem__(self, key, value):
        if value is not None:
            if not issubclass(value.__class__, VariableModel):
                raise TypeError
        self._variables[key] = value

    def __getitem__(self, key):
        # if key not in self._variables.keys():
        #    raise KeyError
        return self._variables[key]

    def __delitem__(self, key):
        del self._variables[key]
        pass

    def __repr__(self):
        rep = f"GeospaceLab Dataset object <{self.label()}, " + \
              f"starting time: {str(self.dt_fr)}, stopping time: {str(self.dt_to)}>"
        return rep.replace('geospacelab.datahub.sources.', '')

    # def add_variable(self, variable, name=None):
    #     if issubclass(variable, VariableModel):
    #         pass
    #     else:
    #         variable = VariableModel(value=variable, name=name, visual=self.visual)
    #     self[variable.name] = variable
    def exist(self, var_name):
        if var_name in self._variables.keys():
            return True
        else:
            return False

    def remove_variable(self, name):
        del self[name]

    def get_variable_names(self):
        return list(self._variables.keys())

    def _set_default_variables(self, default_variable_names, configured_variables=None):
        if configured_variables is None:
            configured_variables = {}
        for var_name in default_variable_names:
            self.add_variable(var_name, configured_variables=configured_variables)

    @property
    def data_root_dir(self) -> pathlib.Path:
        return self._data_root_dir

    @data_root_dir.setter
    def data_root_dir(self, path):
        self._data_root_dir = pathlib.Path(path)


class DownloaderModel(object):
    """
    Downloader Model for downloading data files from different data sources

    Parameters
    ===========
    :param dt_fr: starting time.
    :type dt_fr: datetime.datetime
    :param dt_to: stopping time.
    :type dt_to: datetime.datetime
    :param data_file_root_dir: the root directory storing the data.
    :type data_file_root_dir: str or pathlib.Path object
    :param done: if the files are downloaded or not [False].
    :type done: bool
    :param force: Force the downloader to download the data files, [True].
    :type force: bool
    :param direct_download: Download the data directly without calling the function download [True].
    :type direct_download: bool
    """

    def __init__(self, dt_fr, dt_to, data_file_root_dir=None, force=True, direct_download=True, **kwargs):

        self.dt_fr = dt_fr
        self.dt_to = dt_to
        self.data_file_root_dir = data_file_root_dir

        self.files = []

        self.force = force
        self.direct_download = direct_download
        self.done = False

        if self.direct_download:
            self.done = self.download(**kwargs)

    def download(self, **kwargs):
        """
        Download the data from a http or ftp server. Implementation must be added according to different data sources
        :param kwargs: keywords for downloading the data files.
        """
        raise NotImplemented
