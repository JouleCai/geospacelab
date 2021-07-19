import h5py
import datetime
import pathlib
from pathlib import Path, PurePath
import numpy as np
import pickle
import pathlib

from geospacelab.config.preferences import *
from geospacelab.datahub import LoaderBase
from geospacelab.datahub.sources.madrigal.eiscat import madrigal_eiscat_downloader as downloader

import geospacelab.toolbox.utilities.pydatetime as dttool
import geospacelab.toolbox.utilities.pylogging as pylog

if __name__ == "__main__":
    dir_root = datahub_data_root_dir

default_variable_names = [
    'GB_DATETIME', 'GB_DATETIME_1', 'GB_DATETIME_2',
    'magic_constant', 'half_scattering_angle',
    'az', 'el', 'Tx_power', 'alt', 'range',
    'n_e', 'T_i', 'T_e', 'nu_i', 'v_i_los', 'comp_mix', 'comp_O_p',
    'n_e_err', 'T_i_err', 'T_e_err', 'nu_i_err', 'v_i_los_err', 'comp_mix_err', 'comp_O_p_err',
    'status', 'residual'
]


class Loader(LoaderBase):
    database = 'madrigal'
    facility = 'eiscat'
    def __init__(self, **kwargs):
        super().__init__()

        kwargs.setdefault('datasource_path', datahub_data_root_dir / 'madrigal' / 'eiscat')
        self.site = ''
        self.site_location = []
        self.antenna = ''
        self.experiment = ''
        self.modulation = ''
        self.scan_mode = ''
        self.pulse_code = ''
        self.file_patterns = []
        self.file_ext = ''
        self.save_pickle = False
        self.dates = []
        self.variables = {}
        # initialize vairables
        for vn in default_variable_names:
            self.variables.setdefault(vn, None)

        self.config(**kwargs)
        self.datasource_path = Path(self.datasource_path)

        if list(self.file_names):
            self.mode = 'assigned'

        if self.mode != 'assigned':
            self._search_files()

        if self.file_ext == 'hdf5' and 'eiscat' in self.file_patterns:
            self._load_eiscat_hdf5()
        elif self.file_ext == 'hdf5' and 'mad' in self.file_patterns:
            self._load_madrigal_hdf5()
        elif self.file_ext == 'mat':
            self._load_eiscat_mat()
        elif self.file_ext == 'pickle':
            self._load_pickle()

    def _search_files(self):
        dt_fr = self.dt_fr
        dt_to = self.dt_to
        diff_days = dttool.get_diff_days(dt_fr, dt_to)
        day0 = dttool.get_start_of_the_day(dt_fr)
        for i in range(diff_days + 1):
            thisday = day0 + datetime.timedelta(days=i)
            if self.mode == 'dialog':
                import tkinter as tk
                from tkinter import ttk
                from tkinter import filedialog
                from tkinter.messagebox import showinfo
                # create the root window
                root = tk.tk()
                root.withdraw()
                filetypes = (('eiscat files', '*.' + self.file_ext),('all files', '*.*'))
                filename = filedialog.askopenfilename(
                    title='open a file on' + thisday.strftime('%y-%m-%d'),
                    initialdir = self.datasource_path
                )
                filename = Path(filename)
                self.file_paths.append(filename.parent)
                self.file_names.append(filename.name)
            elif self.mode == 'auto':
                key1 = self.site
                key2 = self.antenna
                if key2 == 'uhf':
                    key1 = 'uhf'
                if key2 == 'vhf':
                    key2 = 'vhf'

                file_path = self.datasource_path / thisday.strftime('%Y') / '_'.join((key1, thisday.strftime('%Y-%m-%d')))

                file_name = '*' + '*'.join(self.file_patterns) + '*.' + self.file_ext
                files = file_path.glob(file_name)
                if len(files) == 0 and self.download:
                    pylog.StreamLogger.warning(
                        "The data file on %s may have not been downloaded!", thisday.strftime("%Y%m%d"))
                    if self.download:
                        pylog.StreamLogger.info("Calling downloader ...")
                        downloadObj = downloader.Downloader(dt_fr=dt_fr, dt_to=dt_to)
                    else:
                        pylog.StreamLogger.info("Try to download the data using download=True.")
                files = file_path.glob(file_name)
                if len(files) == 0:
                    raise FileExistsError

                self.file_paths.append(files[0].parent)
                self.file_names.append(files[0].name)
                self.dates.append(thisday)

    def _load_eiscat_hdf5(self):
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
        # az el tx_power t1 t2 ran height lat lon n_e ti te coll vel ',...
        # 'ne_err ti_err te_err coll_err vel_err stat resid comp r_m0 ',...
        # 'name_site r_XMITloc r_RECloc r_SCangle name_expr r_ver dates starttime stoptime']
        var_info_list = [
            ['az', 'az', 'par1d'],
            ['el', 'el', 'par1d'],
            ['Pt', 'Pt', 'par1d'],
            ['h', 'h', 'par2d'],
            ['range', 'range', 'par2d'],
            ['n_e', 'Ne', 'par2d'],
            ['T_i', 'Ti', 'par2d'],
            ['T_r', 'Tr', 'par2d'],
            ['nu_i', 'Collf', 'par2d'],
            ['v_i_los', 'Vi', 'par2d'],
            ['comp_mix', 'pm', 'par2d'],
            ['comp_O_p', 'po+', 'par2d'],
            ['n_e_err', 'var_Ne', 'par2d'],
            ['T_i_err', 'var_Ti', 'par2d'],
            ['T_r_err', 'var_Tr', 'par2d'],
            ['nu_i_err', 'var_Collf', 'par2d'],
            ['v_i_los_err', 'var_Vi', 'par2d'],
            ['comp_mix_err', 'var_pm', 'par2d'],
            ['comp_O_p_err', 'var_po+', 'par2d'],
            ['status', 'status', 'par2d'],
            ['residual', 'res1', 'par2d']
        ]
        vars = {}
        for ind_f, file_name in enumerate(self.file_names):
            with h5py.File(self.file_paths[ind_f] / file_name, 'r') as fh5:
                h5_data = fh5['data']
                h5_metadata = fh5['metadata']
                num_gates = h5_data['par0d'][15]
                num_times = len(h5_data['utime'][0, :])
                for var_info in var_info_list:
                    var_ind = search_variable(fh5, var_info[1], var_info[2])
                    var_name = var_info[0]
                    var = np.array(h5_data[var_info[2]][var_ind])
                    nrow = num_times
                    ncol = var.size / nrow
                    var = var.reshape(nrow, ncol)
                    vars




    def _load_madrigal_hdf5(self):
        raise NotImplemented

    def _load_eiscat_mat(self):
        raise NotImplemented

    def _load_pickle(self):
        raise NotImplemented

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



