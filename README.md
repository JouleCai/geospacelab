# geospacelab
A python-based library to collect, manage, analyze, and visualize geospace data.

## Features
- With a user-friendly data manager ("Datahub"):
    + Dock to multiple sourced or temporary datasets.
    + Control the data downloading, loading and managing a secondary products.
    + Load data from file with various data formats (hdf, mat, sav, cdf, netcdf, ascii, and binary)
    + Assign variables for visulization.
    + Save variables and attributes to several kinds of data file formats (hdf, mat, or cdf)
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
### 1. The python distribution _Anaconda_ is recommended:
The package was tested with the anaconda distribution and with python>=3.7 under Ubuntu 20.04 and MacOS Big Sur.

With Anaconda, it may be easier to install some required dependencies listed below, e.g., h5py, netcdf4, and cartopy, using the _conda_ command.

It's also recommended to install the package and dependencies in a virtual environment with anaconda. To create a virtual environment, use:

```shell
conda create --name [YOUR_ENV_NAME] python=3.8
```

After creating the virtual environement, you need to activate the virtual environment:

```shell
conda activate [YOUR_ENV_NAME]
```

More detailed instroduction to work with the anaconda environment can be found [here](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#), 

### 2. Required dependencies
The current release requires pre-installation for some dependencies. If you are in an annaconda virtual environment, install the dependencies using the commands below:

```shell
pip install toml 
pip install scipy 
pip install numpy 
pip install matplotlib 
pip install madrigalweb 
pip install aacgmv2 
conda install h5py 
conda install netcdf4 
conda install cartopy 
```
For other python distribution, please check the instroduction of the packages from their own webpages.

### 3. Install and update the package
Install the package via:

```shell
pip install geospacelab
```

Update the package via:
```shell
pip install geospacelab --upgrade
```

Note: The package is currently pre-released. The installation methods may be changed in the future.

### 4. Configuration for the first-time use
Information will be displayed in the python console when you import the package at the first time. You will need to set the root directory for storing the data, and other configurations, e.g., the cookies for some online database. A user's configuration file will be created at:

```
[your_home_directory]/.geospacelab/config.toml
```

You can set your own preferences there. Otherwise, the package will use the default settings. 

To access the madrigal database, it requires the user's full name, email, and affiliation to download the data. The user can set those parameters from python console when call the madrigal modules at the first time. Or, you can add or change the content below in the configuration file "config.toml":

```toml
[datahub.madrigal]
user_fullname = "YOUR_NAME"
user_email = "YOU_EMAIL"
user_affiliation = "YOUR_AFFILIATION"
```

### 5. Uninstallation
Uninstall the package via:
```shell
pip uninstall geospacelab
```
If you don't need the user's configuration, delete the file at **_[your_home_directory]/.geospacelab/config.toml_**

## Usage

### Example 1: EISCAT quicklook plot
The package can download and load EISCAT data automatically from [the EISCAT schedule page](https://portal.eiscat.se/schedule/) with the preset loading mode "AUTO" and file type "eiscat-hdf5". In addition, the package can load data by assigning the data file paths. See introductions in the documentation.

The EISCAT quicklook plot shows the GUISDAP analysed results in the same format as the online EISCAT quicklook plot.
The figure layout and quality are improved. In addition, several marking tools like vertical lines, shadings, top bars can be 
added in the plot. See the example script and figure below:

> example.py
> ```python
> import datetime
> import geospacelab.visualization.eiscat_viewer as eiscat
> from geospacelab import preferences as pfr
> 
> dt_fr = datetime.datetime.strptime('20201209' + '1800', '%Y%m%d%H%M')
> dt_to = datetime.datetime.strptime('20201210' + '0600', '%Y%m%d%H%M')
>
> site = 'UHF'
> antenna = 'UHF'
> modulation = '60'
> load_mode = 'AUTO'
> viewer = eiscat.quicklook(dt_fr, dt_to, site=site, antenna=antenna, modulation=modulation, load_mode='AUTO')
> 
> viewer.save_figure()
> viewer.show()
> ```
>
> Several marking tools can be added as the overlays on the top of the quicklook plot:
> ```python
> # add vertical line
> dt_fr_2 = datetime.datetime.strptime('20201209' + '2030', "%Y%m%d%H%M")
> dt_to_2 = datetime.datetime.strptime('20201210' + '0130', "%Y%m%d%H%M")
> viewer.add_vertical_line(dt_fr_2, bottom_extend=0, top_extend=0.02, label='Line 1', label_position='top')
> # add shading
> viewer.add_shading(dt_fr_2, dt_to_2, bottom_extend=0, top_extend=0.02, label='Shading 1', label_position='top')
> # add top bar
> dt_fr_3 = datetime.datetime.strptime('20201210' + '0130', "%Y%m%d%H%M")
> dt_to_3 = datetime.datetime.strptime('20201210' + '0430', "%Y%m%d%H%M")
> viewer.add_top_bar(dt_fr_3, dt_to_3, bottom=0., top=0.02, label='Top bar 1')
>
> # save figure
> viewer.save_figure()
> # show on screen
> viewer.show()
> ```
> Output:
> ![alt text](https://github.com/JouleCai/geospacelab/blob/master/examples/EISCAT_UHF_beata_cp1_2.1u_CP_20201209-180000-20201210-060000.png?raw=true)

## Notes
- The current version is a pre-released version. Many features will be added soon.
- The full documentation has not been added.

