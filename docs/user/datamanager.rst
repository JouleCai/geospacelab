.. py:currentmodule:: geospacelab.datahub

.. topic:: Overview

    The module datahub is the data manager in GeospaceLab, including three class-based core components:

    - :class:`DataHub <geospacelab.datahub.DataHub>` manages a set of datasets docked or added to the datahub.
    - :class:`Dataset <geospacelab.datahub.DatasetModel>` manages a set of variables loaded from a data source.
    - | :class:`Variable <geospacelab.datahub.VariableModel>` records the value, error, and various attributes
      | (e.g., name, label, unit, depends, ndim, ...) of a variable.

Datahub
^^^^^^^^

To create a DataHub object, either call the function :func:`create_datahub <geospacelab.datahub.create_datahub>` or
the class :class:`DataHub <geospacelab.datahub.DataHub>`. The former provides an option (``datahub_class``)
to create based a DataHub subclass.

.. autofunction:: geospacelab.datahub::create_datahub


.. autoclass:: geospacelab.datahub::DataHub
    :members:
    :inherited-members:

Dataset
^^^^^^^^

All the datasets added to :class:`DataHub` are the objects of :class:`DatasetModel` or its subclasses.
:class:`DatasetModel` is the base class, providing the essential attributes and methods to manage a data source. See details below:

.. autoclass:: geospacelab.datahub.DatasetModel
    :members:
    :inherited-members:

Variable
^^^^^^^^

.. autoclass:: geospacelab.datahub.VariableModel
    :members:
    :inherited-members:

