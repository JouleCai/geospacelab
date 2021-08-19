import datetime
import numpy as np
import requests
import bs4
import pathlib
import re
import netCDF4 as nc
import ftplib
from contextlib import closing

import geospacelab.toolbox.utilities.pydatetime as dttool
import geospacelab.toolbox.utilities.pylogging as mylog
import geospacelab.datahub.sources.wdc as wdc
from geospacelab import preferences as prf


class Downloader(object):

    def __init__(self, dt_fr,  dt_to, data_file_root_dir=None):

        self.dt_fr = dt_fr
        self.dt_to = dt_to

        self.done = False
        if data_file_root_dir is None:
            self.data_file_root_dir = prf.datahub_data_root_dir / 'GFZ' / 'Indices'
        else:
            self.data_file_root_dir = pathlib.Path(data_file_root_dir)

        self.ftp_host = "ftp.gfz-potsdam.de"

        self.download()

    def download(self):
        diff_years = self.dt_to.year - self.dt_fr.year

        for i in range(diff_years + 1):
            dt1 = datetime.datetime(self.dt_fr.year + i, 1, 1)

            ystr = dt1.strftime('%Y')

            with closing(ftplib.FTP()) as ftp:
                try:
                    ftp.connect(self.ftp_host, 21, 30)  # 30 timeout
                    ftp.login()
                    ftp.cwd('/pub/home/obs/Kp_ap_Ap_SN_F107')
                    file_name = 'Kp_ap_Ap_SN_F107_' + ystr + '.txt'
                    file_path = self.data_file_root_dir / file_name
                    if file_path.is_file():
                        mylog.simpleinfo.info(
                            "The file {} exists in the directory {}.".format(file_path.name, file_path.parent.resolve()))
                        self.done = True
                        # continue
                    else:
                        file_path.parent.resolve().mkdir(parents=True, exist_ok=True)
                    with open(file_path, 'w+b') as f:
                        res = ftp.retrbinary('RETR ' + file_name, f.write)

                        if not res.startswith('226 Transfer complete'):
                            print('Downloaded of file {0} is not compile.'.format(file_name))
                            pathlib.Path.unlink(file_path)
                            self.done = False
                            return None
                except:
                    print('Error during download from FTP')
                    self.done = False
                    return

            mylog.StreamLogger.info("Preparing to save the data in the netcdf format ...")
            self.save_to_netcdf(ystr, file_path)

    def save_to_netcdf(self, ystr, file_path):
        with open(file_path, 'r') as f:
            text = f.read()

            results = re.findall(
                r'^(\d+ \d+ \d+)\s*\d+\s*[\d.]+\s*(\d+)\s*(\d+)\s*' +
                r'([\-\d.]+)\s*([\-\d.]+)\s*([\-\d.]+)\s*([\-\d.]+)\s*([\-\d.]+)\s*([\-\d.]+)\s*([\-\d.]+)\s*([\-\d.]+)\s*' +
                r'([\-\d]+)\s*([\-\d]+)\s*([\-\d]+)\s*([\-\d]+)\s*([\-\d]+)\s*([\-\d]+)\s*([\-\d]+)\s*([\-\d]+)\s*([\-\d]+)\s*' +
                r'([\-\d]+)\s*([\-\d.]+)\s*([\-\d.]+)\s*([\-\d]+)',
                text,
                re.M
            )
            results = list(zip(*results))
            dts = [datetime.datetime.strptime(dtstr, "%Y %m %d") for dtstr in results[0]]

            time_array = np.array([(dt - datetime.datetime(1970, 1, 1))/datetime.timedelta(seconds=1) for dt in dts])
            bsr_array = np.array(results[1])
            bsr_array.astype(np.int32)
            db_array = np.array(results[2]).astype(np.int32)
            kp_array = np.array(results[3:11]).astype(np.float32)
            kp_array = np.where(kp_array == -1, np.nan, kp_array)
            ap_array = np.array(results[11:20]).astype(np.float32)
            ap_array = np.where(ap_array == -1, np.nan, ap_array)
            sn_array = np.array(results[20]).astype(np.int32)
            sn_array = np.where(sn_array == -1, np.nan, sn_array)
            f107o_array = np.array(results[21]).astype(np.float32)
            f107o_array = np.where(f107o_array == -1, np.nan, f107o_array)
            f107a_array = np.array(results[22]).astype(np.float32)
            f107a_array = np.where(f107a_array == -1, np.nan, f107a_array)
            flag_array = np.array(results[23]).astype(np.int32)

            num_rows = len(results[0])

            ################## for SN, f10.7
            fp = file_path.parent.resolve() / "SN_F107" / ("GFZ_SN_F107_" + ystr + '.nc')
            fp.parent.resolve().mkdir(parents=True, exist_ok=True)
            fnc = nc.Dataset(fp, 'w')
            fnc.createDimension('UNIX_TIME', num_rows)

            fnc.title = "GFZ SN/F10.7 index"
            time = fnc.createVariable('UNIX_TIME', np.float32, ('UNIX_TIME',))
            time.units = 'Unix Time since 1970-1-1'
            f107o = fnc.createVariable('F107_OBS', np.float32, ('UNIX_TIME',))
            f107a = fnc.createVariable('F107_ADJ', np.float32, ('UNIX_TIME',))
            sn = fnc.createVariable('SN', np.float32, ('UNIX_TIME',))
            bsr = fnc.createVariable('BSRN', np.float32, ('UNIX_TIME',))
            db = fnc.createVariable('BSRN_Days', np.float32, ('UNIX_TIME',))
            flag = fnc.createVariable('Flag', np.float32, ('UNIX_TIME',))
            time[::] = time_array[::]
            f107o[::] = f107o_array[::]
            f107a[::] = f107a_array[::]
            sn[::] = sn_array[::]
            bsr[::] = bsr_array[::]
            db[::] = db_array[::]
            flag_1 = np.where(flag_array < 2, 0, 1)
            flag[::] = flag_1[::]
            print('From {} to {}.'.format(
            datetime.datetime.utcfromtimestamp(time_array[0]),
            datetime.datetime.utcfromtimestamp(time_array[-1]))
            )
            mylog.StreamLogger.info(
                "The requested SN/F10.7 data has been downloaded and saved in the file {}.".format(fp))
            fnc.close()

            ########## for Kp Ap
            fp = file_path.parent.resolve() / "Kp_Ap" / ("GFZ_Kp_Ap_" + ystr + '.nc')
            fp.parent.resolve().mkdir(parents=True, exist_ok=True)
            fnc = nc.Dataset(fp, 'w')
            fnc.createDimension('UNIX_TIME', num_rows*8)

            fnc.title = "GFZ SN/F10.7 index"
            time = fnc.createVariable('UNIX_TIME', np.float32, ('UNIX_TIME',))
            kp = fnc.createVariable('Kp', np.float32, ('UNIX_TIME',))
            ap = fnc.createVariable('ap', np.float32, ('UNIX_TIME',))
            Ap = fnc.createVariable('Ap', np.float32, ('UNIX_TIME',))
            flag = fnc.createVariable('flag', np.float32, ('UNIX_TIME',))
            seconds = np.arange(8) * 3600 * 3 + 1800*3
            time_array = np.tile(time_array, (8, 1)).T
            for i in range(8):
                time_array[:, i] = time_array[:, i] + seconds[i]
            time_array = time_array.flatten()
            kp_array = np.round(kp_array.T.flatten(), decimals=1)
            Ap_array = np.tile(ap_array[-1, :].flatten(), (8, 1)).T.flatten()
            ap_array = ap_array[:-1, :].T.flatten()
            flag_array = np.tile(flag_array, (8, 1)).T.flatten()
            flag_2 = np.where(flag_array == 2, 1, 0)
            time[::] = time_array[::]
            kp[::] = kp_array[::]
            ap[::] = ap_array[::]
            Ap[::] = Ap_array[::]
            flag[::] = flag_2[::]
            print('From {} to {}.'.format(
            datetime.datetime.utcfromtimestamp(time_array[0]),
            datetime.datetime.utcfromtimestamp(time_array[-1]))
            )
            mylog.StreamLogger.info(
                "The requested Kp/Ap data has been downloaded and saved in the file {}.".format(fp))
            fnc.close()

            self.done = True


if __name__ == "__main__":
    dt_fr1 = datetime.datetime(1990, 1, 1)
    dt_to1 = datetime.datetime(2020, 12, 16)
    Downloader(dt_fr1, dt_to1)



