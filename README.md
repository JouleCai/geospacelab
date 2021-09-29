# GeospaceLab (geospacelab)
[![License](./docs/images/license-badge.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Python](./docs/images/python-badge.svg)](https://www.python.org/) 
[![DOI](https://zenodo.org/badge/347315860.svg)](https://zenodo.org/badge/latestdoi/347315860)


To collect, manage, and visualize geospace data in an easy and fast way. The documentation can be found 
on [readthedocs.io](https://geospacelab.readthedocs.io/en/latest/).

## Features
- Class-based data manager, including
  - __DataHub__: the core module to manage data from multiple sources,
  - __DatasetModel__: the base class to download, load, and process data from a data source, 
  - __VariableModel__: the base class to store the value, error, and other attributes for a variable.
- Extendable
  - Provide a standard procedure from downloading, loading, and post-processing the data.
  - Easy to extend for a data source which has not been included in the package.
  - Flexible to add functions for post-processing.
- Visualization
  - Time series plots with 
    - automatically adjustable time ticks and tick labels.
    - dynamical panels (easily adding or removing panels).
    - useful marking tools (vertical line crossing panels, shadings, top bars, etc, see Example 2 in
[Usage](https://github.com/JouleCai/geospacelab#usage))
  - Map projection
    - Polar views with
      - coastlines in either GEO or AACGM (APEX) coordinate system.
      - mapping in either fixed lon/mlon mode or in fixed LST/MLT mode.
    - Support 1-D or 2-D plots with
      - satellite tracks (time ticks and labels)
      - nadir colored 1-D plots
      - gridded surface plots 
- Space coordinate system transformation
- Toolboxes for data analysis
  - Basic toolboxes for numpy array, datetime, logging, python dict, list, and class.
  - Coordinate system transformation.

## Built-in data sources:
| Data Source                   | Variables             | File Format           | Downloadable  | Express         | Status      | 
|-------------------------------|-----------------------|-----------------------|---------------|-----------------|-------------|
| CDAWeb/OMNI                   | Solar wind and IMF    |*cdf*                 | *True*        | __OMNIViewer__  | stable      |
| Madrigal/EISCAT               | Ionospheric Ne, Te, Ti, ... | *EISCAT-hdf5*, *Madrigal-hdf5* | *True* | __EISCATViewer__ | stable    |
| Madrigal/GNSS/TECMAP          | Ionospheric GPS TEC map | *hdf5*                | *True*        | -  | beta      |
| Madrigal/DMSP/s1              | DMSP SSM, SSIES, etc  | *hdf5*                | *True*        | __DMSPTSViewer__  | beta      |
| Madrigal/DMSP/s4              | DMSP SSIES            | *hdf5*                | *True*        | __DMSPTSViewer__  | beta      |
| Madrigal/DMSP/e               | DMSP SSJ              | *hdf5*                | *True*        | __DMSPTSViewer__  | beta      |
| JHUAPL/DMSP/SSUSI             | DMSP SSUSI            | *netcdf*              | *True*        | __DMSPSSUSIViewer__  | beta      |
| JHUAPL/AMPERE/fitted          | AMPERE FAC            | *netcdf*              | *False*        | __AMPEREViewer__  | stable      |
| WDC/Dst                       | Dst index             | *IAGA2002-ASCII*      | *True*        | - | stable |
| WDC/ASYSYM                    | ASY/SYM indices       | *IAGA2002-ASCII*      | *True*        | __OMNIViewer__ | stable |
| WDC/AE                        | AE indices            | *IAGA2002-ASCII*      | *True*        | __OMNIViewer__ | stable |
| GFZ/Kp                        | Kp/Ap indices         | *ASCII*               | *True*        | -              | beta   |
| GFZ/SNF107                    | SN, F107              | *ASCII*               | *True*        | -              | beta   |
| ESA/SWARM/EFI_LP_1B           | SWARM Ne, Te, etc.    | *netcdf*              | *True*        | -              | beta   |
| ESA/SWARM/AOB_FAC_2F          | SWARM FAC, auroral oval boundary | *netcdf*              | *True*        | -              | beta   |
| UTA/GITM/2DALL                | GITM 2D output        | *binary*, *IDL-sav*   | *False*       | -              | beta   |
| UTA/GITM/3DALL                | GITM 3D output        | *binary*, *IDL-sav*   | *False*       | -              | beta   |



## Installation
### 1. The python distribution "*__Anaconda__*" is recommended:
The package was tested with the anaconda distribution and with **PYTHON>=3.7** under **Ubuntu 20.04** and **MacOS Big Sur**.

With Anaconda, it may be easier to install some required dependencies listed below, e.g., cartopy, using the _conda_ command.
It's also recommended installing the package and dependencies in a virtual environment with anaconda. 

After [installing the anaconda distribution](https://docs.anaconda.com/anaconda/install/index.html), a virtual environment can be created by the code below in the terminal:

```shell
conda create --name [YOUR_ENV_NAME] python=3.8 spyder
```
The package "spyder" is a widely-used python IDE. Other IDEs, like "VS Code" or "Pycharm" also work.

After creating the virtual environement, you need to activate the virtual environment:

```shell
conda activate [YOUR_ENV_NAME]
```
and then to install the package as shown below or to start the IDE "spyder".

More detailed information to set the anaconda environment can be found [here](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#), 

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

### 5. Upgrade

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

In "example1.py":
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
# assign a variable from its own dataset to the datahub
n_e = dh.assign_variable('n_e')
T_i = dh.assign_variable('T_i')

# get the variables which have been assigned in the datahub
n_e = dh.get_variable('n_e')
T_i = dh.get_variable('T_i')
# if the variable is not assigned in the datahub, but exists in the its own dataset:
comp_O_p = dh.get_variable('comp_O_p', dataset=ds_1)     # O+ ratio
# above line is equivalent to
comp_O_p = dh.datasets[1]['comp_O_p']

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

In "example2.py"
```python
import datetime
import geospacelab.express.eiscat_viewer as eiscat

dt_fr = datetime.datetime.strptime('20201209' + '1800', '%Y%m%d%H%M')
dt_to = datetime.datetime.strptime('20201210' + '0600', '%Y%m%d%H%M')

# check the eiscat-hdf5 filename from the EISCAT schedule page, e.g., "EISCAT_2020-12-10_beata_60@uhfa.hdf5"
site = 'UHF'
antenna = 'UHF'
modulation = '60'
load_mode = 'AUTO'
# The code will download and load the data automatically as long as the parameters above are set correctly.
viewer = eiscat.EISCATViewer(
      dt_fr, dt_to, site=site, antenna=antenna, modulation=modulation, load_mode='AUTO'
)
viewer.quicklook()
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

### Example 3: OMNI and WDC geomagnetic indices:

In "example3.py"
```python
import datetime
import geospacelab.express.omni_viewer as omni

dt_fr = datetime.datetime.strptime('20160314' + '0600', '%Y%m%d%H%M')
dt_to = datetime.datetime.strptime('20160320' + '0600', '%Y%m%d%H%M')

omni_type = 'OMNI2'
omni_res = '1min'
load_mode = 'AUTO'
viewer = omni.OMNIViewer(
    dt_fr, dt_to, omni_type=omni_type, omni_res=omni_res, load_mode=load_mode
)
viewer.quicklook()

# data can be retrieved in the same way as in Example 1:
viewer.list_assigned_variables()
B_x_gsm = viewer.get_variable('B_x_GSM')
# save figure
viewer.save_figure()
# show on screen
viewer.show()
```
Output:
> ![alt text](https://github.com/JouleCai/geospacelab/blob/master/examples/OMNI_1min_20160314-060000-20160320-060000.png?raw=true)

## Citation and Acknowledgements
Please acknowledge or cite GeospaceLab, if the library contributes to a project that leads
to a scientific publication.


## Notes
- The current version is a pre-released version. Many features will be added soon.


