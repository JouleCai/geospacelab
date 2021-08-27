import netCDF4
# from cdf.internal import EPOCHbreakdown
import os
import datetime
import numpy as np

import ssusi_read_edr
import ssusi_download
# from davitpy import utils


if __name__ == "__main__":

    dir_root = os.getcwd()
    dir_ssusi = "/media/lcai/My Passport/KTH-Backup/01_work/00_data/JHUAPL/DMSP/SSUSI/"
    date_in = datetime.datetime( 2015,9, 8 ) # input date
    id_sat = "f16" # satellite number
    id_orbit = '%05d' % 61353 # orbit number
    id_pole = 'N' # northern (N) pole or southern (S) pole

    opt_ssusi_sdr = {'dir'          : dir_ssusi,    \
                     'date'         : date_in,      \
                     'satID'        : id_sat,       \
                     'orbitID'      : id_orbit,     \
                     'datatype'     : 'edr-aur',        \
                     'UVband'       : ['LBHL', 'LBHS', '1216'],     \
                     'pole'         : id_pole
                     }
    readObj = ssusi_read_edr.ProcessData(opt = opt_ssusi_sdr)

    readObj.load_request_data()

    # if hasattr(readObj, 'pole'):
    #    readObj.filter_data_pole(boundinglat = 25)


class ProcessData(object):

    def __init__(self, opt = None):

        if opt is None:
            print("Error: presettings are required!")

        self.filepath = opt['dir'] + opt['satID'] + "/" \
                        + opt['date'].strftime("%Y%m%d/")
        self.date = opt['date']
        self.satID = opt['satID']
        self.orbitID = opt['orbitID']
        self.datatype = opt['datatype']
        self.UVband = opt['UVband']
        self.pole = opt['pole']
        for root, dirs, files in os.walk(self.filepath):
            for fNum, fName in enumerate(files):
                if self.orbitID in fName:
                    if "EDR-AUR" in fName:
                        self.filename = fName

        if not hasattr(self, 'filename'):
            downloadObj = ssusi_download.SSUSIDownload(outBaseDir=self.filepath)
            downloadObj.download_files(self.date, [self.datatype],  \
                                       self.satID, self.orbitID)
            self.filename = downloadObj.filename
#        self.UVBand = UVBand


    def load_request_data(self):
        dataset = netCDF4.Dataset(self.filepath + self.filename)

        pdict = {}
        pdict['SC_LAT'] = np.array(dataset.variables['LATITUDE'])
        pdict['SC_LON'] = np.array(dataset.variables['LONGITUDE'])
        pdict['SC_ALT'] = np.array(dataset.variables['ALTITUDE'])

        if self.pole == 'N':
            pole_str = 'NORTH'
        else:
            pole_str = 'SOUTH'
        pdict['MLAT'] = np.array(dataset.variables['LATITUDE_GEOMAGNETIC_GRID_MAP'])
        pdict['MLON'] = np.array(dataset.variables['LONGITUDE_GEOMAGNETIC_' + pole_str + '_GRID_MAP'])
        pdict['MLT'] = np.array(dataset.variables['MLT_GRID_MAP'])
        pdict['UT'] = np.array(dataset.variables['UT_' + self.pole])

        aob_mlat = np.array(dataset.variables[pole_str + '_GEOMAGNETIC_LATITUDE'])[:, np.newaxis]
        aob_mlon = np.array(dataset.variables[pole_str + '_GEOMAGNETIC_LONGITUDE'])[:, np.newaxis]
        aob_mlt = np.array(dataset.variables[pole_str + '_MAGNETIC_LOCAL_TIME'])[:, np.newaxis]
        pdict['AOB_EQ'] = np.concatenate((aob_mlat, aob_mlon, aob_mlt), axis=1)

        aob_mlat = np.array(dataset.variables[pole_str + '_POLAR_GEOMAGNETIC_LATITUDE'])[:, np.newaxis]
        aob_mlon = np.array(dataset.variables[pole_str + '_POLAR_GEOMAGNETIC_LONGITUDE'])[:, np.newaxis]
        aob_mlt = np.array(dataset.variables[pole_str + '_POLAR_MAGNETIC_LOCAL_TIME'])[:, np.newaxis]
        pdict['AOB_PO'] = np.concatenate((aob_mlat, aob_mlon, aob_mlt), axis=1)

        aob_mlat = np.array(dataset.variables['MODEL_' + pole_str + '_GEOMAGNETIC_LATITUDE'])[:, np.newaxis]
        aob_mlon = np.array(dataset.variables['MODEL_' + pole_str + '_GEOMAGNETIC_LONGITUDE'])[:, np.newaxis]
        aob_mlt = np.array(dataset.variables['MODEL_' + pole_str + '_MAGNETIC_LOCAL_TIME'])[:, np.newaxis]
        pdict['MAOB_EQ'] = np.concatenate((aob_mlat, aob_mlon, aob_mlt), axis=1)

        aob_mlat = np.array(dataset.variables['MODEL_' + pole_str + '_POLAR_GEOMAGNETIC_LATITUDE'])[:, np.newaxis]
        aob_mlon = np.array(dataset.variables['MODEL_' + pole_str + '_POLAR_GEOMAGNETIC_LONGITUDE'])[:, np.newaxis]
        aob_mlt = np.array(dataset.variables['MODEL_' + pole_str + '_POLAR_MAGNETIC_LOCAL_TIME'])[:, np.newaxis]
        pdict['MAOB_PO'] = np.concatenate((aob_mlat, aob_mlon, aob_mlt), axis=1)
        for uvb in self.UVband:
            if uvb == '1216':
                uvb_id = 0
            elif uvb == '1304':
                uvb_id = 1
            elif uvb == '1356':
                uvb_id = 2
            elif uvb == 'LBHS':
                uvb_id = 3
            elif uvb == 'LBHL':
                uvb_id = 4

            pdict[uvb] = np.array(dataset.variables['DISK_RADIANCEDATA_INTENSITY_' + pole_str][uvb_id, :, :])

        self.paras = pdict

    def filter_data_pole(self, boundinglat = 25):
        paras = self.paras
        PP_keys = paras.keys()
        boundingS = boundinglat
        boundingN = -boundinglat
        for PP_key in PP_keys:
            data_keys = paras[PP_key].keys()
            SClat = paras[PP_key]['SClat'][0]
            rec_N = np.where(SClat > boundingN)[0] # orbit passing northern hemisphere first!
            rec_S = np.where(SClat < boundingS)[0]

            rec_Nextra = np.where(rec_N > rec_S[0])[0]
            rec_N = rec_N[:rec_Nextra[0]-1]

            if self.pole == 'N':
                rec_pole = rec_N
            elif self.pole == 'S':
                rec_pole = rec_S
            for d_key in data_keys:
                if d_key in ['PPalt']:
                    continue
                paras[PP_key][d_key] = paras[PP_key][d_key][:, rec_pole]
        self.paras = paras