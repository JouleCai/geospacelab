# Licensed under the BSD 3-Clause License
# Copyright (C) 2021 GeospaceLab (geospacelab)
# Author: Lei Cai, Space Physics and Astronomy, University of Oulu

__author__ = "Lei Cai"
__copyright__ = "Copyright 2021, GeospaceLab"
__license__ = "BSD-3-Clause License"
__email__ = "lei.cai@oulu.fi"
__docformat__ = "reStructureText"


import toml
import pathlib
import time
import os
import geospacelab.toolbox.utilities.pylogging as mylog

# datahub_data_root_dir = pathlib.Path("/Users/lcai/01-Work/00-Data/")
# datahub_data_root_dir = pathlib.Path("/home/lei/afys-data")

package_name = "geospacelab"

on_rtd = os.environ.get('READTHEDOCS') == 'True'


class Preferences(object):
    _on_rtd = on_rtd

    def __init__(self):
        self.package_name = package_name

        self.user_config = {}
        self.set_user_config()

        if self._on_rtd:
            self.datahub_data_root_dir = pathlib.Path.home() / 'Geospacelab' / 'Data'
        else:
            try:
                self.datahub_data_root_dir = self.user_config['datahub']['data_root_dir']
            except KeyError:
                self.datahub_data_root_dir = None

    @property
    def datahub_data_root_dir(self):
        return self._datahub_data_root_dir

    @datahub_data_root_dir.setter
    def datahub_data_root_dir(self, path):
        if path == '':
            path = None
        if path is None:
            data_path = pathlib.Path.home() / 'Geospacelab' / 'Data'
            time.sleep(0.2)
            result = input(f"Store the data in the default directory ({data_path})? [y]/n: ")
            time.sleep(0.2)
            if result.lower() in ['', 'y', 'yes']:
                self._datahub_data_root_dir = data_path
            elif result.lower() in ['n', 'no']:
                time.sleep(0.2)
                result = input('Input the root directory for storing data: ')
                time.sleep(0.2)
                if str(result):
                    self._datahub_data_root_dir = pathlib.Path(result)
                else:
                    raise NotADirectoryError('"Set the default root directory in ~/.geospacelab/config.toml!"')
            else:
                raise ValueError("The input is not correct!")
        else:
            self._datahub_data_root_dir = pathlib.Path(path)

        if not self._datahub_data_root_dir.is_dir():
            mylog.simpleinfo.info(f"The direction ({self._datahub_data_root_dir}) does not exist!")
            if self._on_rtd:
                result = 'y'
            else:
                time.sleep(0.2)
                result = input("Create it? [y]/n: ")
                time.sleep(0.2)
            if result.lower() in ['', 'y', 'yes']:
                self._datahub_data_root_dir.mkdir(parents=True, exist_ok=False)
            elif result.lower() in ['n', 'no']:
                mylog.simpleinfo.info("The directory was not created! Set the default root directory in ~/.geospacelab/config.toml!")
            else:
                raise ValueError("The input is not correct!")
        else:
            if path is None:
                mylog.simpleinfo.info(f"Note: The directory ({self._datahub_data_root_dir}) has existed!")

        self.user_config['datahub']['data_root_dir'] = str(self._datahub_data_root_dir)
        if path is None:
            self.set_user_config(user_config=self.user_config, set_as_default=True)

    def set_user_config(self, user_config=None, set_as_default=False):

        home_dir = pathlib.Path.home()

        config_file_path = home_dir / ('.' + package_name) / 'config.toml'

        config_file_path.parent.mkdir(exist_ok=True, parents=True)

        if not config_file_path.is_file():
            config_string = """
# This is a document for the user's configuration in the TOML format.
package_name = \"{}\"
[datahub]
data_root_dir = \"\"
            """.format(package_name)
            mylog.StreamLogger.info(
                "A toml file for user's configuration has been created at {}.".format(config_file_path)
            )
            time.sleep(2)
            config_file_path.touch()
            with open(config_file_path, 'w') as f:
                f.write(config_string)

        user_config_dict = {}
        if user_config is not None:
            if isinstance(user_config, str):
                user_config_dict = toml.loads(user_config)
            elif isinstance(user_config, dict):
                user_config_dict = user_config
            if set_as_default:
                with open(config_file_path, 'w') as f:
                    toml.dump(user_config, f)
        else:
            user_config_dict = toml.load(config_file_path)
        self.user_config = user_config_dict
        return user_config_dict
