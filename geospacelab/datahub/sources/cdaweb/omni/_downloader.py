import datetime
import requests

import geospacelab.toolbox.utilities.pydatetime as dttool
import geospacelab.toolbox.utilities.pylogging as mylog
from geospacelab import preferences as prf


class Downloader(object):
    def __init__(self, dt_fr,  dt_to, res='1min', new_omni=True, data_file_root_dir=None, version='v01'):
        self.dt_fr = dt_fr
        self.dt_to = dt_to
        self.res = res
        self.new_omni = new_omni
        self.version = version
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
        dt0 = datetime.datetime(self.dt_fr.year, self.dt_fr.month, 1)
        for nm in range(num_month):
            dt1 = dttool.get_next_n_months(dt0, nm)
            if self.res in ["1min", "5min"]:
                href = self.base_link + name + '_' + self.res + '/' + '{:4d}'.format(dt1.year) + '/' \
                       + 'omni_' + name + '_' + self.res + '_' + dt1.strftime("%Y%m%d") + '_' + self.version + '.cdf'
            else:
                raise NotImplementedError

            filepath = self.data_file_root_dir / (name + '_' + self.res) / \
                       '{:4d}'.format(dt1.year) / ('omni_' + name + '_' + self.res + '_' + dt1.strftime("%Y%m%d") +
                                                   '_' + self.version + '.cdf')

            filepath.parent.resolve().mkdir(parents=True, exist_ok=True)
            r = requests.get(href, allow_redirects=True)
            with open(filepath, "wb") as omni:
                mylog.simpleinfo.info(
                    "Downloading {} to the directory {} ...".format(filepath.name, filepath.parent.resolve())
                )
                omni.write(r.content)
                mylog.simpleinfo.info("Done")


def test():
    dt_fr = datetime.datetime(2020, 3, 4)
    dt_to = datetime.datetime(2020, 5, 3)
    download_obj = Downloader(dt_fr, dt_to)
    download_obj.download()
    pass


if __name__ == "__main__":
    test()
