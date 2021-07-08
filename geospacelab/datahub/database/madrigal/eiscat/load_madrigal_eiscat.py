import h5py
import datetime
import numpy as np
import pickle

from config.preferences import *
from config.globals import *
from utilities.logging_config import *

from loaders import load_dmsp_madrigal_s as dmsp_read
from downloaders import download_dmsp_madrigalweb as dmsp_download

if __name__ == "__main__":
    dir_root = root_dir_data
    dir_dmsp = dir_root + "/madrigal/DMSP/"
    date_in = datetime.datetime(2015, 10, 31)
    id_sat = "f19"
    id_file = "s4"  # s1: ion velocity; s4: temperature, O+; e: partical energy - use load_dmsp_madrigal_e

    opt_read = {
        'dir':      dir_dmsp,
        'date':     date_in,
        'satID':    id_sat,
        'fileID':   id_file,
        'dtRange':  None,
    }

    # dmsp_read.list_hdf5_structure(None)

    readObj = dmsp_read.ProcessData(opt=opt_read, readObjFile=False,
                                    saveObjFile=False)
    readObj.load_request_data()
    if opt_read['dtRange'] is not None:
        readObj.filter_request_data(opt_read['dtRange'])


class Loader(object):
    def __init__(self, dt_fr, dt_to, save_pickle=True, )



class ProcessData(object):

    def __init__(self, opt=None, readObjFile=False, saveObjFile=False):
        if opt is None:
            return
        if readObjFile:
            filepath_obj = opt['dir'] + opt['date'].strftime("%Y%m%d") + '/'
            filename_obj = 'DMSP_' + opt['satID'].upper() + '_' \
                           + opt['date'].strftime("%Y%m%d") + '_madrigal_' \
                           + opt['fileID'] + '.pkl'
            self.saveObj = {
                'filepath':     filepath_obj,
                'filename':     filename_obj,
                'saveObj':      saveObjFile
            }

        self.date = opt['date']
        self.satID = opt['satID']
        self.fileID = opt['fileID']
        self.filepath = opt['dir'] + self.date.strftime("%Y%m%d") + "/"
        filekey = "_" + self.satID[1:] + self.fileID
        isfile = self.isFile(filekey, self.filepath, filetype="hdf5")
        if not isfile:
            StreamLogger.warning("No data available!")
            StreamLogger.info("Calling downloader ...")
            dt_start = self.date
            dt_stop = self.date
            downloadObj = dmsp_download.DownloadProcess(
                dt_start=dt_start,
                dt_stop=dt_stop
            )
            downloadObj.download_files(filename_keys=[self.fileID], root_dir=root_dir_data)
            isfile = self.isFile(filekey, self.filepath, filetype="hdf5")
            if not isfile:
                StreamLogger.warning("Data do not exist!")

    def isFile(self, filekey, filepath, filetype = None):
        for root, dirs, files in os.walk(filepath):
            for fNum, fName in enumerate(files):
                if filetype is not None:
                    if "." + filetype != os.path.splitext(fName)[1]:
                        continue

                if filekey in fName:
                    self.filename = fName
                    return True
        return False

    def load_request_data(self):

        if hasattr(self, 'saveObj'):
            if not self.saveObj['saveObj']:
                filepath_obj = self.saveObj['filepath']
                filename_obj = self.saveObj['filename']
                if os.path.isfile(filepath_obj + filename_obj):
                    with open(filepath_obj + filename_obj, 'rb') as fparas:
                        paras = pickle.load(fparas)
                        self.paras = paras
                        return
                else:
                    self.saveObj['saveObj'] = True

        with h5py.File(self.filepath + self.filename, 'r') as fh5:

            # show data file structure

            data_params = ['YEAR', 'MONTH', 'DAY', 'HOUR', 'MIN', 'SEC', \
                           'GDLAT', 'GLON', 'GDALT', 'MLT', 'MLAT', 'MLONG', \
                           'SAT_ID', 'NE', 'HOR_ION_V', 'VERT_ION_V', \
                           'BD', 'B_FORWARD', 'B_PERP', \
                           'DIFF_BD', 'DIFF_B_FOR', 'DIFF_B_PERP']
            # Split out the data
            tablecolnames = fh5["Metadata"]["Data Parameters"][:]
            datatable = fh5["Data"]["Table Layout"][:]

            # Make dictionary to translate between coluumn names and numbers
            param_name_to_colnum = {param[0].decode("utf-8"): colind
                                    for colind, param in enumerate(tablecolnames)}
            data_params = list(param_name_to_colnum)
            colinds = [param_name_to_colnum[param_name] \
                       for param_name in data_params]

            nrows = datatable.shape[0]
            ncols = len(data_params)

            data = np.zeros((nrows, ncols))
            data.fill(np.nan)

            maxrowind = 0
            for rowind, row in enumerate(datatable):
                data[rowind, :] = [row[colind] for colind in colinds]
                maxrowind = rowind
                if rowind == 0:
                    for icolind, colind in enumerate(colinds):
                        simpleinfo.info("First value from column %d, name %s is %f",
                                        colind, data_params[icolind], row[colind])


            self.paras = {}

            for pid, paraname in enumerate(data_params):
                if paraname in ['YEAR', 'MONTH', 'DAY', 'HOUR', 'MIN', 'SEC']:
                    continue
                self.paras[paraname] = np.array(data[:maxrowind, pid]).reshape(maxrowind, 1)

            dtlist = np.empty((0, 1))
            for tid in range(maxrowind):
                yy = int(data[tid, 0])
                mm = int(data[tid, 1])
                dd = int(data[tid, 2])
                HH = int(data[tid, 3])
                MM = int(data[tid, 4])
                SS = int(data[tid, 5])
                dt = datetime.datetime(yy, mm, dd, HH, MM, SS)
                dtlist = np.vstack((dtlist, np.array(dt).reshape(1, 1)))

            self.paras['datetime'] = dtlist
            dt_delta = dtlist - self.date
            sectime = np.array([dt_temp.total_seconds() \
                                for dt_temp in dt_delta[:, 0]])
            self.paras['sectime'] = sectime.reshape(maxrowind, 1)

            if hasattr(self, 'saveObj'):
                if self.saveObj['saveObj']:
                    filepath_obj = self.saveObj['filepath']
                    filename_obj = self.saveObj['filename']
                    with open(filepath_obj + filename_obj, 'wb') as fparas:
                        pickle.dump(self.paras, fparas, pickle.HIGHEST_PROTOCOL)

    def filter_request_data(self, dtRange):
        dtRange = np.array(dtRange)
        dt_delta = dtRange - self.date
        secRange = np.array([dt_temp.total_seconds() \
                             for dt_temp in dt_delta])
        para_keys = self.paras.keys()
        seclist = self.paras['sectime'][:, 0]
        ind_dt = np.where((seclist >= secRange[0]) & (seclist <= secRange[1]))[0]
        for pkey in para_keys:
            self.paras[pkey] = self.paras[pkey][ind_dt, :]


def list_hdf5_structure(fh5):
    # example: /home/lcai/01_work/SPADAViewer/data/madrigal/DMSP/20151014/dms_20151014_16e.001.hdf5
    if fh5 is None:
        fn = "/home/lcai/01_work/SPADAViewer/data/madrigal/DMSP/20151031/dms_20151031_16e.002.hdf5"
        fh5 = h5py.File(fn, 'r')

    print(fh5.keys())
    print(fh5['Metadata'].keys())
    print(fh5['Metadata']['Data Parameters'][:])
    print(fh5['Metadata']['Experiment Notes'][:])
    print(fh5['Metadata']['Experiment Parameters'][:])
    print(fh5['Metadata']['Independent Spatial Parameters'][:])
    print(fh5['Metadata']['_record_layout'][:])
    print(fh5['Data'].keys())
    print(fh5['Data']['Array Layout'].keys())  # Note: s4 s1 files do not have 'Array Layout'
    print(fh5['Data']['Array Layout']['1D Parameters'].keys())
    print(fh5['Data']['Array Layout']['1D Parameters']['Data Parameters'][:])
    print(fh5['Data']['Array Layout']['1D Parameters']['el_i_ener'][:])
    print(fh5['Data']['Array Layout']['2D Parameters'].keys())
    print(fh5['Data']['Array Layout']['Layout Description'][:])
    print(fh5['Data']['Array Layout']['ch_energy'][:])
    print(fh5['Data']['Array Layout']['timestamps'][:])
    print(fh5['Data']['Table Layout'][:])





