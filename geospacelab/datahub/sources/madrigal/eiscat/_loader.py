import h5py
import datetime
import pathlib
from pathlib import Path, PurePath
import numpy as np
import pickle
import pathlib

from geospacelab import preferences as prf
# from geospacelab.datahub.sources.madrigal.eiscat import madrigal_eiscat_downloader as downloader

import geospacelab.toolbox.utilities.pydatetime as dttool
import geospacelab.toolbox.utilities.pylogging as pylog
import geospacelab.toolbox.utilities.numpyarray as arraytool
import geospacelab.datahub.sources.madrigal.eiscat as eiscat


default_variable_names = [
    'DATETIME', 'DATETIME_1', 'DATETIME_2',
    'magic_constant', 'r_SCangle', 'r_m0_1', 'r_m0_2'
    'az', 'el', 'P_Tx', 'height', 'range',
    'n_e', 'T_i', 'T_e', 'nu_i', 'v_i_los', 'comp_mix', 'comp_O_p',
    'n_e_err', 'T_i_err', 'T_e_err', 'nu_i_err', 'v_i_los_err', 'comp_mix_err', 'comp_O_p_err',
    'status', 'residual'
]


def select_loader(file_type):
    load_func = None
    if file_type == 'eiscat-hdf5':
        load_func = load_eiscat_hdf5
    elif file_type == 'eiscat-mat':
        load_func = load_eiscat_mat
    elif file_type == 'madrigal-hdf5':
        load_func = load_madrigal_hdf5
    else:
        raise ValueError
    return load_func


def load_eiscat_hdf5(file_path):
    """
    Load EISCAT hdf5 file.
    :param file_paths: a list of the file fullpaths for the hdf5 files
    :return:
        load_obj: instance of Loader
            .variables: queried variables
            .metadata: metadata
    """
    def search_variable(fh5, vn, var_groups=None):
        """
        vn: variable name
        """
        rec = 0
        ind_rec = None
        var_groups_all = ['par0d', 'par0d_sd', 'par1d', 'par1d_sd', 'par2d', 'par2d_pp']
        if var_groups is None:
            var_groups = var_groups_all
        for vg in var_groups:
            var_name_list = fh5['metadata'][vg][:, 0]
            var_name_list = [vn1.decode('UTF-8').strip() for vn1 in var_name_list]
            try:
                ind_rec = var_name_list.index(vn)
                rec = 1
                break
            except ValueError:
                pass

        if rec == 0:
            raise LookupError
        return ind_rec
    # *az *el *tx_power *t1 *t2 *ran *height lat lon *n_e *ti *te *coll *vel ',...
    # '*ne_err *ti_err *te_err *coll_err *vel_err *stat *resid ?comp ?r_m0 ',...
    # 'name_site r_XMITloc r_RECloc r_SCangle name_expr r_ver dates starttime stoptime']
    var_name_dict = {
        'magic_constant': 'Magic_const',
        'r_SCangle': 'SCangle',
        'r_m0_2': 'm02',
        'r_m0_1': 'm01',
        'az': 'az',
        'el': 'el',
        'P_Tx': 'Pt',
        'height': 'h',
        'range': 'range',
        'n_e': 'Ne',
        'T_i': 'Ti',
        'T_r': 'Tr',
        'nu_i': 'Collf',
        'v_i_los': 'Vi',
        'comp_mix': 'pm',
        'comp_O_p': 'po+',
        'n_e_var': 'var_Ne',
        'T_i_var': 'var_Ti',
        'T_r_var': 'var_Tr',
        'nu_i_var': 'var_Collf',
        'v_i_los_var': 'var_Vi',
        'comp_mix_var': 'var_pm',
        'comp_O_p_var': 'var_po+',
        'status': 'status',
        'residual': 'res1'
    }

    site_info = {
        'T': 'UHF',
        'V': 'VHF',
        'K': 'KIR',
        'S': 'SOD',
        'L': 'ESR'
    }

    vars = {}
    metadata = {}

    with h5py.File(file_path, 'r') as fh5:
        var_info_list = eiscat.list_eiscat_hdf5_variables(fh5)
        h5_data = fh5['data']
        h5_metadata = fh5['metadata']
        ind_nrec = var_info_list['name'].index('nrec')
        nrec_group = var_info_list['group'][ind_nrec]
        nrec_group_ind = var_info_list['index'][ind_nrec]
        nrec = h5_data[nrec_group][nrec_group_ind]
        num_row = h5_data['utime'][0].shape[0]
        for var_name, var_name_h5 in var_name_dict.items():
            ind_v = var_info_list['name'].index(var_name_h5)
            var_group = var_info_list['group'][ind_v]
            var_ind = var_info_list['index'][ind_v]
            var = h5_data[var_group][var_ind]
            if var_group == 'par0d':
                metadata[var_name] = var[0]
            elif var_group == 'par1d':
                var = var.reshape(num_row, 1)
                vars[var_name] = var
            elif var_group == 'par2d':
                if nrec_group == 'par0d':
                    num_col = int(var.shape[0] / num_row)
                    if num_col != nrec[0]:
                        print("Note: the number of range gates doesn't match nrec!")
                    var = var.reshape(num_row, num_col)
                elif nrec_group == 'par1d':
                    num_gates = int(np.max(nrec))
                    var_array = np.empty((num_row, num_gates))
                    var_array[:, :] = np.nan
                    rec_ind_1 = 0
                    for i in range(num_row):
                        rec_ind_2 = int(rec_ind_1 + nrec[i])
                        var_array[i, :int(nrec[i])] = var[rec_ind_1:rec_ind_2]
                        rec_ind_1 = rec_ind_2
                    var = var_array
                vars[var_name] = var

        # unix time to datetime
        utime1 = h5_data['utime'][0]
        dt1 = dttool.convert_unix_time_to_datetime(utime1)
        var = dt1.reshape(num_row, 1)
        var_name = 'DATETIME_1'
        vars[var_name] = var

        utime2 = h5_data['utime'][1]
        dt2 = dttool.convert_unix_time_to_datetime(utime2)
        var = dt2.reshape(num_row, 1)
        var_name = 'DATETIME_2'
        vars[var_name] = var

        vars['DATETIME'] = vars['DATETIME_1'] + (vars['DATETIME_2'] - vars['DATETIME_2']) / 2

        metadata['r_XMITloc'] = [h5_data['par0d'][2][0], h5_data['par0d'][3][0], h5_data['par0d'][4][0]]
        metadata['r_RECloc'] = [h5_data['par0d'][5][0], h5_data['par0d'][6][0], h5_data['par0d'][7][0]]
        metadata['site_name'] = site_info[h5_metadata['names'][1][1].decode('UTF-8').strip()]
        metadata['pulse_code'] = h5_metadata['names'][0][1].decode('UTF-8').strip()
        metadata['antenna'] = h5_metadata['names'][2][1].decode('UTF-8').strip()
        metadata['GUISDAP_version'] = h5_metadata['software']['GUISDAP_ver'][0, 0].decode('UTF-8').strip()
        metadata['rawdata_path'] = h5_metadata['software']['gfd']['data_path'][0, 0].decode('UTF-8').strip()
        metadata['scan_mode'] = metadata['rawdata_path'].split('_')[1]
        metadata['affiliation'] = metadata['rawdata_path'].split('@')[0].split('_')[-1]

    vars_add = {}
    for var_name, value in vars.items():
        if '_var' in var_name:
            var_name_err = var_name.replace('_var', '_err')
            vars_add[var_name_err] = np.sqrt(vars[var_name])
    vars.update(vars_add)

    vars['DATETIME'] = vars['DATETIME_1'] + (vars['DATETIME_2'] - vars['DATETIME_1'])/2
    vars['T_e'] = vars['T_i'] * vars['T_r']
    vars['T_e_err'] = vars['T_e'] * np.sqrt((vars['T_i_err']/vars['T_i'])**2
                                            + (vars['T_r_err']/vars['T_r'])**2)
    vars['height'] = vars['height'] / 1000.
    vars['range'] = vars['range'] / 1000.

    if 'az' in metadata.keys():
        az = np.empty((vars['DATETIME_1'].shape[0], 1))
        az[:, :] = vars['az']
        vars['az'] = az
    if 'el' in metadata.keys():
        el = np.empty((vars['DATETIME_1'].shape[0], 1))
        el[:, :] = vars['el']
        vars['el'] = el
    vars['az'] = np.mod(vars['az'], 360)
    vars['el'] = np.mod(vars['el'], 360)

    load_obj = Loader(vars, metadata)
    return load_obj


def load_eiscat_mat():
    raise NotImplemented


def load_madrigal_hdf5():
    raise NotImplemented


class Loader:
    def __init__(self, vars, metadata):
        self.variables = vars
        self.metadata = metadata


if __name__ == "__main__":
    dir_root = prf.datahub_data_root_dir

    filepath = pathlib.Path('examples') / "EISCAT_2005-09-01_steffe_64@32m.hdf5"

    load_eiscat_hdf5([filepath])

#
# class Loader(LoaderBase):
#     database = 'madrigal'
#     facility = 'eiscat'
#     def __init__(self, **kwargs):
#         super().__init__()
#
#         kwargs.setdefault('datasource_path', datahub_data_root_dir / 'madrigal' / 'eiscat')
#         self.site = ''
#         self.site_location = []
#         self.antenna = ''
#         self.experiment = ''
#         self.modulation = ''
#         self.scan_mode = ''
#         self.pulse_code = ''
#         self.file_patterns = []
#         self.file_ext = ''
#         self.save_pickle = False
#         self.dates = []
#         self.variables = {}
#         # initialize vairables
#         for vn in default_variable_names:
#             self.variables.setdefault(vn, None)
#
#         self.config(**kwargs)
#         self.datasource_path = Path(self.datasource_path)
#
#         if list(self.file_names):
#             self.mode = 'assigned'
#
#         if self.mode != 'assigned':
#             self._search_files()
#
#         if self.file_ext == 'hdf5' and 'eiscat' in self.file_patterns:
#             self._load_eiscat_hdf5()
#         elif self.file_ext == 'hdf5' and 'mad' in self.file_patterns:
#             self._load_madrigal_hdf5()
#         elif self.file_ext == 'mat':
#             self._load_eiscat_mat()
#         elif self.file_ext == 'pickle':
#             self._load_pickle()
#
#     def _search_files(self):
#         dt_fr = self.dt_fr
#         dt_to = self.dt_to
#         diff_days = dttool.get_diff_days(dt_fr, dt_to)
#         day0 = dttool.get_start_of_the_day(dt_fr)
#         for i in range(diff_days + 1):
#             thisday = day0 + datetime.timedelta(days=i)
#             if self.mode == 'dialog':
#                 import tkinter as tk
#                 from tkinter import ttk
#                 from tkinter import filedialog
#                 from tkinter.messagebox import showinfo
#                 # create the root window
#                 root = tk.tk()
#                 root.withdraw()
#                 filetypes = (('eiscat files', '*.' + self.file_ext),('all files', '*.*'))
#                 filename = filedialog.askopenfilename(
#                     title='open a file on' + thisday.strftime('%y-%m-%d'),
#                     initialdir = self.datasource_path
#                 )
#                 filename = Path(filename)
#                 self.file_paths.append(filename.parent)
#                 self.file_names.append(filename.name)
#             elif self.mode == 'auto':
#                 key1 = self.site
#                 key2 = self.antenna
#                 if key2 == 'uhf':
#                     key1 = 'uhf'
#                 if key2 == 'vhf':
#                     key2 = 'vhf'
#
#                 file_path = self.datasource_path / thisday.strftime('%Y') / '_'.join((key1, thisday.strftime('%Y-%m-%d')))
#
#                 file_name = '*' + '*'.join(self.file_patterns) + '*.' + self.file_ext
#                 files = file_path.glob(file_name)
#                 if len(files) == 0 and self.download:
#                     pylog.StreamLogger.warning(
#                         "The data file on %s may have not been downloaded!", thisday.strftime("%Y%m%d"))
#                     if self.download:
#                         pylog.StreamLogger.info("Calling downloader ...")
#                         downloadObj = downloader.Downloader(dt_fr=dt_fr, dt_to=dt_to)
#                     else:
#                         pylog.StreamLogger.info("Try to download the data using download=True.")
#                 files = file_path.glob(file_name)
#                 if len(files) == 0:
#                     raise FileExistsError
#
#                 self.file_paths.append(files[0].parent)
#                 self.file_names.append(files[0].name)
#                 self.dates.append(thisday)
#
#     def _load_eiscat_hdf5(self):
#         def search_variable(fh5, vn, var_groups=None):
#             """
#             vn: variable name
#             """
#             rec = 0
#             ind_rec = None
#             var_groups_all = ['par0d', 'par0d_sd', 'par1d', 'par1d_sd', 'par2d', 'par2d_pp']
#             if var_groups is None:
#                 var_groups = var_groups_all
#             for vg in var_groups:
#                 var_name_list = fh5['metadata'][vg][:, 0]
#                 var_name_list = [vn1.decode('UTF-8').strip() for vn1 in var_name_list]
#                 try:
#                     ind_rec = var_name_list.index(vn)
#                     rec = 1
#                     break
#                 except ValueError:
#                     pass
#
#             if rec == 0:
#                 raise LookupError
#             return ind_rec
#         # az el tx_power t1 t2 ran height lat lon n_e ti te coll vel ',...
#         # 'ne_err ti_err te_err coll_err vel_err stat resid comp r_m0 ',...
#         # 'name_site r_XMITloc r_RECloc r_SCangle name_expr r_ver dates starttime stoptime']
#         var_info_list = [
#             ['az', 'az', 'par1d'],
#             ['el', 'el', 'par1d'],
#             ['Pt', 'Pt', 'par1d'],
#             ['h', 'h', 'par2d'],
#             ['range', 'range', 'par2d'],
#             ['n_e', 'Ne', 'par2d'],
#             ['T_i', 'Ti', 'par2d'],
#             ['T_r', 'Tr', 'par2d'],
#             ['nu_i', 'Collf', 'par2d'],
#             ['v_i_los', 'Vi', 'par2d'],
#             ['comp_mix', 'pm', 'par2d'],
#             ['comp_O_p', 'po+', 'par2d'],
#             ['n_e_err', 'var_Ne', 'par2d'],
#             ['T_i_err', 'var_Ti', 'par2d'],
#             ['T_r_err', 'var_Tr', 'par2d'],
#             ['nu_i_err', 'var_Collf', 'par2d'],
#             ['v_i_los_err', 'var_Vi', 'par2d'],
#             ['comp_mix_err', 'var_pm', 'par2d'],
#             ['comp_O_p_err', 'var_po+', 'par2d'],
#             ['status', 'status', 'par2d'],
#             ['residual', 'res1', 'par2d']
#         ]
#         vars = {}
#         for ind_f, file_name in enumerate(self.file_names):
#             with h5py.File(self.file_paths[ind_f] / file_name, 'r') as fh5:
#                 h5_data = fh5['data']
#                 h5_metadata = fh5['metadata']
#                 num_gates = h5_data['par0d'][15]
#                 num_times = len(h5_data['utime'][0, :])
#                 for var_info in var_info_list:
#                     var_ind = search_variable(fh5, var_info[1], var_info[2])
#                     var_name = var_info[0]
#                     var = np.array(h5_data[var_info[2]][var_ind])
#                     nrow = num_times
#                     ncol = var.size / nrow
#                     var = var.reshape(nrow, ncol)
#                     vars
#
#
#
#
#     def _load_madrigal_hdf5(self):
#         raise NotImplemented
#
#     def _load_eiscat_mat(self):
#         raise NotImplemented
#
#     def _load_pickle(self):
#         raise NotImplemented
#
#     def filter_request_data(self, dtRange):
#         dtRange = np.array(dtRange)
#         dt_delta = dtRange - self.date
#         secRange = np.array([dt_temp.total_seconds() \
#                              for dt_temp in dt_delta])
#         para_keys = self.paras.keys()
#         seclist = self.paras['sectime'][:, 0]
#         ind_dt = np.where((seclist >= secRange[0]) & (seclist <= secRange[1]))[0]
#         for pkey in para_keys:
#             self.paras[pkey] = self.paras[pkey][ind_dt, :]
#
#

