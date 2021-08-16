import datetime
import requests

import geospacelab.toolbox.utilities.pydatetime as dttool
from geospacelab import preferences as prf


class Downloader(object):
    def __init__(self, dt_fr,  dt_to, res='1min', new_omni=True, data_file_root_dir=None):
        self.dt_fr = dt_fr
        self.dt_to = dt_to
        self.res = res
        self.new_omni = new_omni
        if data_file_root_dir is None:
            self.data_file_root_dir = prf.datahub_data_root_dir / "CDAWeb" / "OMNI"
        else:
            self.data_file_root_dir = data_file_root_dir

        self.base_link = "https://cdaweb.gsfc.nasa.gov/pub/data/omni/omni_cdaweb/"

    def download(self):
        if self.new_omni:
            name = "hro2"
        else:
            name = "hro"

        num_month = (self.dt_to.year - self.dt_fr.year) * 12 + self.dt_to.month - self.dt_fr.month + 1
        dt0 = datetime.datetime(self.dt_fr.year, self.dt_fr.minth, 1)
        for nm in range(num_month):
            dt0 = dttool.get_next_n_months(dt0, nm)
            if self.res in ["1min", "5min"]:
                href = self.base_link + name + '_' + self.res + '/' + '{:4d}'.format(dt0.year) + '/' \
                       + 'omni_' + name + '_' + self.res + '_' + dt0.strftime("%Y%m%d") + '_v01.cdf'
            else:
                raise NotImplementedError

            filepath = self.data_file_root_dir / (name + '_' + self.res) / '{:4d}'.format(dt0.year) / \
                'omni_' + name + '_' + self.res + '_' + dt0.strftime("%Y%m%d") + '_v01.cdf'

            r = requests.get(href, allow_redirects=True)
            with open(filepath, "wb") as omni:
                omni.write(r.content)
