.. _dev:

Guidance for developers
=======================
This chapter will give a guidance for advanced users and developers, who want to
contribute to the development of GeospaceLAB by 
maintaining existing code, adding new code, or improving documentation.


Maintainence of existing code
-----------------------------
If you find any bugs, redundant code, or other errors in the source code and example code, 
you can first submit a report to Issues_. If you have fixed the bugs or errors in your local 
repository, you can create a pull request for the changed script.


.. _Issues: https://github.com/JouleCai/geospacelab/issues


Contribution of new code
-------------------------
Contributing new code includes submitting a new sourced dataset 
and adding new functionality for the existing classes and modules.


Submitting new sourced dataset
^^^^^^^^^^^^^^^^^^^^
GeospaceLAB aims to implement automatically downloading, storing, and loading data
from pupolar online databases. To achieve this goal, it requires several stragies to
build the specialized subclass of SourcedDataset and associated classes and modules such
as Loader, Downloader and variable configuration.

For a skillful developer, it is not difficult to refer the classes for 
the existing sourced datasets in GeospaceLAB. The source code of the existing sourced datasets
are store in `DataHub <https://github.com/JouleCai/geospacelab/tree/master/geospacelab/datahub/sources>`__.
In that directory, the listed folders are the main categories of a sourced dataset, indicating
the data providers, e.g., madrigal for the Madrigal database. A main catogory can have nested subfolders,
indicating the levels of sub-catogories. 
The suggested levels of the nested folders can be like this:
    [database] -> [facility] -> [instrument] -> [data product]
However, it is not neccessary to always organize the nested folders in above way. Some of the levels
can be omitted. GeospaceLAB will search all the dataset classes from those folders recursively. The levels
are kinds of guidelines for the developer to organize the sourced datasets.

The modules for a specific sourced dataset can be found in a last-level sub-folder. For example,
the modules for OMNI dataset are stored `here <https://github.com/JouleCai/geospacelab/tree/master/geospacelab/datahub/sources/cdaweb/omni>`__
The folder includes four files: *__init__.py*, *loader.py*, *downloader.py*, and 'variable_config.py'.
The four files are also typically included for other sourced datasets.
First, the file *__init__.py* includes a SourcedDataset subclass with global settings.
Second, the file *loader.py* includes a Loader class for loading the data from the local folder.
Third, the file *downloader.py* includes a Downloader class for downloading the files from the
online database to the local folder. 
Finally, the file *variable_config* inlcludes the pre-settings for several key variables included 
in the dataset. 
The dataset class in *__init__.py* will communicate with other classes to implement the goal mentioned
in the beginning. 

To build a new sourced dataset, please refer to above folders and files. One can use the exisiting
classess as a template or subclass them if the proposed dataset has many similarities with the existing one.


Adding new functionality
^^^^^^^^^^^^^^^^^^
If you have been working with GeospaceLAB for a while, you probably have some associated code
that can become useful for others. Maybe it is a good idea to contribute it to GeospaceLAB to
extend the functionality in some classes or modules. Then you could create a pull request for 
the new code.

At the current stage, several features are prioritized, including specialized methods in the
existing sourced datasets, new dashboards in the module Express, new plotting methods in the 
module Visualization for 1-D and 2-D data, and subclasses of GeoPanel with new map projections.






