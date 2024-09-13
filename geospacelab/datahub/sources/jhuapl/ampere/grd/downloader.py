import re
import numpy
import pathlib

import geospacelab.toolbox.utilities.pydatetime as dttool
import geospacelab.datahub.sources.jhuapl.ampere.downloader as downloader
import geospacelab.datahub.sources.jhuapl.ampere as ampere


class Downloader(downloader.Downloader):


    def __init__(self, dt_fr, dt_to,
            data_file_root_dir=None,
            data_product='grd', pole='N', user_name=ampere.default_user_name, direct_download=True, force_download=False):

        super().__init__(dt_fr, dt_to, 
                         data_file_root_dir=data_file_root_dir,
                         data_product=data_product, pole=pole, user_name=user_name,
                         direct_download=direct_download, force_download=force_download)        


