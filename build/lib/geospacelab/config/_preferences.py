

import pathlib

# datahub_data_root_dir = pathlib.Path("/Users/lcai/01-Work/00-Data/")
# datahub_data_root_dir = pathlib.Path("/home/lei/afys-data")


class Preferences(object):

    def __init__(self):
        self.package_name = "geospacelab"
        self.datahub_data_root_dir = None

    @property
    def datahub_data_root_dir(self):
        return self._datahub_data_root_dir

    @datahub_data_root_dir.setter
    def datahub_data_root_dir(self, path):
        if path is None:
            self._datahub_data_root_dir = pathlib.Path.home() / 'Geospacelab_Data'
        else:
            self._datahub_data_root_dir = pathlib.Path(path)

