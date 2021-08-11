# geospacelab
A python-based library to collect, manage, and visualize geospace data.

## Features
- With a user-friendly data manager ("Datahub"):
    + Dock with multiple datasets (sourced or temporary).
    + Control to download, load, and manage the data.
    + Load data from files with formats (hdf, mat, sav, cdf, netcdf, ascii, and binary)
    + Assign variables for visualization.
    + Give variable type and attributes.
- Provide utilities for data analysis.
- Useful visualization components based on "matplotlib" and "cartopy".
    + Time series plots with:
        - Automatically adjustable time ticks.
        - Marking tools including vertical lines, shadings, top bars, etc...
        - Data gap removing.
    + Map projections.
- Add or remove panels by a simple configuration ("panel_layouts")
- Produce publication-ready plots.

## Installation
### 1. The python distribution "*__Anaconda__*" is recommended:
The package was tested with the anaconda distribution and with python>=3.7 under Ubuntu 20.04 and MacOS Big Sur.

With Anaconda, it may be easier to install some required dependencies listed below, e.g., cartopy, using the _conda_ command.

It's also recommended to install the package and dependencies in a virtual environment with anaconda. To create a virtual environment, use:

```shell
conda create --name [YOUR_ENV_NAME] python=3.8
```

After creating the virtual environement, you need to activate the virtual environment:

```shell
conda activate [YOUR_ENV_NAME]
```

More detailed instroduction to work with the anaconda environment can be found [here](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#), 

### 2. Installation
#### Quick install from the pre-built release (recommended):
```shell
pip install geospacelab
```

#### Install from [Github](https://github.com/JouleCai/geospacelab) (not recommended):
```shell
pip install git+https://github.com/JouleCai/geospacelab@master
```

### 2. Dependencies
The package dependencies need to be installed before or after the installation of the package. 
Several dependencies will be installed automatically with the package installation, 
including __toml__, __requests__, __bueatifulsoup4__, __numpy__, __scipy__, __matplotlib__, __h5py__, __netcdf4__,
__cdflib__, __madrigalweb__, and __aacgmv2__.

Other dependencies will be needed if you see a *__ImportError__* or *__ModuleNotFoundError__* 
displayed in the python console. Some frequently used modules and their installation methods are listed below:
- [__cartopy__](https://scitools.org.uk/cartopy/docs/latest/installing.html): Map projection for geospatial data.
  - ```conda install -c conda-forge cartopy ``` 
- [__apexpy__ \*](https://apexpy.readthedocs.io/en/latest/reference/Apex.html): Apex and Quasi-Dipole geomagnetic 
coordinate system. 
  - ```pip install apexpy ```
- [__geopack__](https://github.com/tsssss/geopack): The geopack and Tsyganenko models in Python.
  - ```pip install geopack ```

> ([\*]()): The **_gcc_** or **_gfortran_** compilers are required before installing the package. 
> - MacOS: ```brew install gcc ``` 
> - Linux: ```sudo apt install gcc gfortran ```  

Please install the packages above, if needed.

Note: The package is currently pre-released. The installation methods may be changed in the future.


### 4. First-time startup and basic configuration
Some basic configurations will be made with the first-time import of the package. Following the messages prompted in the python console, the first configuration is to set the root directory for storing the data.

When the modules to access the online Madrigal database is imported,  it will ask for the inputs of user's full name, email, and affiliation.

The user's configuration can be found from the *__toml__* file below:
```
[your_home_directory]/.geospacelab/config.toml
```
The user can set or change the preferences in the configuration file. For example, to change the root directory for storing the data, modify or add the lines in "config.toml":
```toml
[datahub]
data_root_dir = "YOUR_ROOT_DIR"
```
To set the Madrigal cookies, change the lines:
```toml
[datahub.madrigal]
user_fullname = "YOUR_NAME"
user_email = "YOU_EMAIL"
user_affiliation = "YOUR_AFFILIATION"
```

### 5. Update

If the package is installed from the pre-built release. Update the package via:
```shell
pip install geospacelab --upgrade
```

### 6. Uninstallation
Uninstall the package via:
```shell
pip uninstall geospacelab
```
If you don't need the user's configuration, delete the file at **_[your_home_directory]/.geospacelab/config.toml_**

## Usage
### Example 1: Dock a sourced dataset and get variables:
The core of the data manager is the class Datahub. A Datahub instance will be used for docking a buit-in sourced dataset, or adding a temporary or user-defined dataset. 

The "dataset" is a Dataset instance, which is used for loading and downloading 
the data. 

Here is an example to load the EISCAT data from the online service.  The module will download EISCAT data automatically from 
[the EISCAT schedule page](https://portal.eiscat.se/schedule/) with the preset loading mode "AUTO" and file type "eiscat-hdf5". 
In addition, the package can load data by assigning the data file paths.

```python
import datetime

from geospacelab.datahub import DataHub

# settings
dt_fr = datetime.datetime.strptime('20210309' + '0000', '%Y%m%d%H%M')   # datetime from
dt_to = datetime.datetime.strptime('20210309' + '2359', '%Y%m%d%H%M')   # datetime to
database_name = 'madrigal'      # built-in sourced database name 
facility_name = 'eiscat'        # facility name

site = 'UHF'                # facility attributes required, check from the eiscat schedule page
antenna = 'UHF'
modulation = 'ant'

# create a datahub instance
dh = DataHub(dt_fr, dt_to)
# dock a dataset
ds_1 = dh.dock(datasource_contents=[database_name, facility_name],
                      site=site, antenna=antenna, modulation=modulation, data_file_type='eiscat-hdf5')
# load data
ds_1.load_data()
# get the variables which have been assigned in the dataset 
n_e = dh.get_variable('n_e', dataset=ds_1) # equivalent to n_e = ds_1['n_e'], return a Variable instance.
# the variable will be retrieved from the latest added dataset, if dataset is not specified 
T_i = dh.get_variable('T_i') # equivalent to T_i = ds_1['T_i']   
# The variables, e.g., n_e and T_i, are the class Variable's instances, 
# which stores the variable values, errors, and many other attributes, e.g., name, label, unit, depends, ....
# To get the value of the variable, use variable_isntance.value, e.g.,
print(n_e.value)        # return the variable's value, type: numpy.ndarray, axis 0 is always along the time, check n_e.depends.items{}
print(n_e.error)

```

### Example 2: EISCAT quicklook plot
The EISCAT quicklook plot shows the GUISDAP analysed results in the same format as the online EISCAT quicklook plot.
The figure layout and quality are improved. In addition, several marking tools like vertical lines, shadings, top bars can be 
added in the plot. See the example script and figure below:

```python
import datetime
import geospacelab.visualization.eiscat_viewer as eiscat

dt_fr = datetime.datetime.strptime('20201209' + '1800', '%Y%m%d%H%M')
dt_to = datetime.datetime.strptime('20201210' + '0600', '%Y%m%d%H%M')

site = 'UHF'
antenna = 'UHF'
modulation = '60'
load_mode = 'AUTO'
viewer = eiscat.quicklook(
      dt_fr, dt_to, site=site, antenna=antenna, modulation=modulation, load_mode='AUTO'
)

# viewer.save_figure() # comment this if you need to run the following codes
# viewer.show()   # comment this if you need to run the following codes.

"""
The viewer is an instance of the class EISCATViewer, which is a heritage of the class Datahub.
Thus, the variables can be retrieved in the same ways as shown in Example 1. 
"""
n_e = viewer.get_variable('n_e')

"""
Several marking tools (vertical lines, shadings, and top bars) can be added as the overlays 
on the top of the quicklook plot.
"""
# add vertical line
dt_fr_2 = datetime.datetime.strptime('20201209' + '2030', "%Y%m%d%H%M")
dt_to_2 = datetime.datetime.strptime('20201210' + '0130', "%Y%m%d%H%M")
viewer.add_vertical_line(dt_fr_2, bottom_extend=0, top_extend=0.02, label='Line 1', label_position='top')
# add shading
viewer.add_shading(dt_fr_2, dt_to_2, bottom_extend=0, top_extend=0.02, label='Shading 1', label_position='top')
# add top bar
dt_fr_3 = datetime.datetime.strptime('20201210' + '0130', "%Y%m%d%H%M")
dt_to_3 = datetime.datetime.strptime('20201210' + '0430', "%Y%m%d%H%M")
viewer.add_top_bar(dt_fr_3, dt_to_3, bottom=0., top=0.02, label='Top bar 1')

# save figure
viewer.save_figure()
# show on screen
viewer.show()
```
Output:
> ![alt text](https://github.com/JouleCai/geospacelab/blob/master/examples/EISCAT_UHF_beata_cp1_2.1u_CP_20201209-180000-20201210-060000.png?raw=true)

## Notes
- The current version is a pre-released version. Many features will be added soon.
- The full documentation has not been added.

