Installation
==============

Python environment
-------------------
The python distribution `Anaconda <https://www.anaconda.com/>`_ is recommended. The package was tested with the anaconda distribution
with Python>=3.7 under Ubuntu 20.04 and MacOS Big Sur. Some of the dependencies are more easily installed by using
the *conda* commands, e.g., `cartopy`_.

It's also recommended to create a **virtual environment** to avoid the version conflicts. To create a virtual environment, try:

    .. code-block:: bash

        conda create --name [YOUR_ENV_NAME] python=3.8 spyder

The package "spyder" is a widely-used python IDE. Other IDEs, like "VS Code" or "Pycharm" also work fine.

Remember to activate the virtual environment, before installing the package and processing your project:

    .. code-block:: bash

        conda activate [YOUR_ENV_NAME]

More detailed information to manage the anaconda environment can be found
`here <https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`_,

Quick installation
----------------------

Install from the pre-built release (recommended):

    .. code-block:: bash

       pip install geospacelab

Install from Github:

    .. code-block:: bash

        pip install git+https://github.com/JouleCai/geospacelab@master

Dependencies
------------
Most of the dependencies will be installed automatically when the package is installed,
including toml, requests, natsort, bueatifulsoup4,
numpy, scipy, matplotlib,
h5py, netcdf4, cdflib, madrigalweb, and aacgmv2.

Some other dependencies will be needed if you see a ImportError or ModuleNotFoundError displayed in the python console.
Those packages are listed below:

- `cartopy`_: Map projection for geospatial data.

    .. code-block:: bash

        conda install -c conda-forge cartopy

- `apexpy <https://github.com/aburrell/apexpy>`_: Apex and Quasi-Dipole geomagnetic coordinate system.

    .. code-block:: bash

        pip install apexpy

    .. note::
        The gcc or gfortran compilers are required before installing "apexpy".

        MacOS: ``brew install gcc``

        Ubuntu: ``sudo apt install gcc gfortran``

Please install the packages above, if needed.


.. _cartopy:  <https://scitools.org.uk/cartopy/docs/latest/installing.html>


First-time startup and basic configuration
------------------------------------------

Some basic configurations will be set when the package is imported for the first time. Please pay attention on the message
prompted in the python console.
First, a root directory for storing the data need to be set. The default root directory is ``[YOUR_HOMR_DIR]/geospacelab/data``.
You are able to set another directory as you want.

Second, some online database may ask inputs on the user's information.
For example, When the modules to access the online Madrigal database is imported, the package will ask for the inputs of
user's full name, email, and affiliation.

All the user's configuration can be reset by editing the toml file at
``[YOUR_HOME_DIR]/.geospacelab/config.toml``.

For example, to change the root directory for storing the data, modify or add the lines in the file "config.toml":

    .. code-block:: toml

        [datahub]
        data_root_dir = "YOUR_ROOT_DIR"

To set the Madrigal cookies, change the lines:

    .. code-block:: toml

        [datahub.madrigal]
        user_fullname = "YOUR_NAME"
        user_email = "YOU_EMAIL"
        user_affiliation = "YOUR_AFFILIATION"


Upgrade
-------

If the package is installed from the pre-built release. Update the package via:

    .. code-block:: bash

        pip install geospacelab --upgrade

Uninstallation
--------------

Uninstall the package via:

    .. code-block:: bash

        pip uninstall geospacelab