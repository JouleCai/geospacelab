# Licensed under the BSD 3-Clause License
# Copyright (C) 2021 GeospaceLab (geospacelab)
# Author: Lei Cai, Space Physics and Astronomy, University of Oulu

__author__ = "Lei Cai"
__copyright__ = "Copyright 2021, GeospaceLab"
__license__ = "BSD-3-Clause License"
__email__ = "lei.cai@oulu.fi"
__docformat__ = "reStructureText"

# from geospacelab.datahub.sources.esa_eo.swarm.loader import LoaderModel

# define the default variable name dictionary
default_variable_name_dict = {
    'n_e': 'NE',
    'v_i_H': 'HOR_ION_V',
    'V_i_V': 'VERT_ION_V',
    'B_D': 'BD',
    'B_F': 'B_FORWARD',
    'B_P': 'B_PERP',
    'B_D_DIFF': 'DIFF_BD',
    'B_F_DIFF': 'DIFF_B_FOR',
    'B_P_DIFF': 'DIFF_B_PERP',
    'SC_GEO_LAT': 'GDLAT',
    'SC_GEO_LON': 'GLON',
    'SC_GEO_ALT': 'GDALT',
    'SC_MAG_LAT': 'MLAT',
    'SC_MAG_LON': 'MLONG',
    'SC_MAG_MLT': 'MLT',
}



import pathlib
import h5py
class Loader(object):
    """
    :param file_path: the full path of the data file
    :type file_path: pathlib.Path or str
    :param file_type: the type of the file, [cdf]
    :param variable_name_dict: the dictionary for mapping the variable names from the cdf files to the dataset
    :type variable_name_dict: dict
    :param direct_load: call the method :meth:`~.LoadModel.load_data` directly or not
    :type direct_load: bool
    """
    def __init__(self, file_path, file_type='madrigal-hdf5', variable_name_dict=None, direct_load=True, **kwargs):

        self.file_path = pathlib.Path(file_path)
        self.file_type = file_type
        self.variables = {}

        if variable_name_dict is None:
            variable_name_dict = default_variable_name_dict
        self.variable_name_dict = variable_name_dict

        if direct_load:
            self.load_data()

    def load_data(self, **kwargs):
        if self.file_type == 'madirgal-hdf5':
            self.load_hdf5_data()

    def load_hdf5_data(self):
        """
        load the data from the cdf file
        :return:
        """
        with h5py.File(self.file_path, 'r') as fh5:
            # load metadata
            metadata = {}

            exp_params = fh5['Metadata']['Experiment Parameters'][:]
            exp_params = list(zip(*tuple(exp_params)))
            fn_id = exp_params[0].index(b'Cedar file name')
            fn = exp_params[1][fn_id].decode('UTF-8')
            rc = re.compile(r'^MAD[\w]+_[\d]{4}-[\d]{2}-[\d]{2}_(\w+)@(\w+)\.hdf5')
            rm = rc.findall(fn)[0]
            pattern_1 = rm[0]
            pattern_2 = rm[1]
            metadata['modulation'] = pattern_1.split('_')[-1]
            metadata['pulse_code'] = pattern_1.replace('_' + metadata['modulation'], '')

            antenna = pattern_2
            if 'uhf' in antenna:
                sitename = 'UHF'
            elif 'vhf' in antenna:
                sitename = 'VHF'
            elif 'sod' in antenna:
                sitename = 'SOD'
            elif 'kir' in antenna:
                sitename = 'KIR'
            elif '32' in antenna or '42' in antenna:
                sitename = 'ESR'
            else:
                print(antenna)
                raise AttributeError
            metadata['site_name'] = sitename

            lat_id = exp_params[0].index(b'instrument latitude')
            lat = float(exp_params[1][lat_id].decode('UTF-8'))
            lon_id = exp_params[0].index(b'instrument longitude')
            lon = float(exp_params[1][lon_id].decode('UTF-8'))
            alt_id = exp_params[0].index(b'instrument altitude')
            alt = float(exp_params[1][alt_id].decode('UTF-8')) * 1000

            metadata['r_RECloc'] = [lat, lon, alt]
            if sitename in ['ESR', 'UHF', 'VHF', 'TRO']:
                metadata['r_XMITloc'] = [lat, lon, alt]
            else:
                metadata['r_XMITloc'] = [69.583, 19.21, 30]
            metadata['antenna'] = antenna
            metadata['GUISDAP_version'] = ''
            metadata['rawdata_path'] = ''
            metadata['scan_mode'] = ''
            metadata['affiliation'] = ''
            metadata['rawdata_path'] = ''
            metadata['scan_mode'] = ''
            metadata['affiliation'] = ''

            # load data
            data = fh5['Data']['Table Layout'][:]
            data = list(zip(*tuple(data)))
            data_parameters = list(zip(*tuple(fh5['Metadata']['Data Parameters'][:])))
            var_names_h5 = [vn.decode('UTF-8') for vn in data_parameters[0]]
            nvar_h5 = len(var_names_h5)
            ran_id = var_names_h5.index('RANGE')
            ran = data[ran_id]
            inds_ran_min = argrelmin(np.array(ran))[0]
            inds_ran_min = np.append(inds_ran_min, 0)
            inds_ran_min.sort()

            inds_ran_max = argrelmax(np.array(ran))[0]
            inds_ran_max = np.append(inds_ran_max, len(ran)-1)
            inds_ran_max.sort()

            ngates_max = np.max(np.diff(inds_ran_max))
            data_array = np.empty((nvar_h5, inds_ran_min.shape[0], ngates_max))
            data_array[::] = np.nan
            for ip in range(nvar_h5):
                var_tmp = np.array(data[ip])
                for i in range(inds_ran_min.shape[0]):
                    ind1 = inds_ran_min[i]
                    ind2 = inds_ran_max[i]
                    data_array[ip, i, 0: ind2-ind1+1] = var_tmp[ind1: ind2+1]
            num_row = inds_ran_min.shape[0]
            vars_h5 = {}
            for ip in range(nvar_h5):

                var = data_array[ip]
                test_unique = np.unique(var[0][~np.isnan(var[0])])
                if test_unique.shape[0] == 1 and not re.match(r'^D\w+', var_names_h5[ip]) and var_names_h5[ip] not in ['GFIT']:
                    # print(var_names_h5[ip])
                    var = np.array([list(set(var[i][~np.isnan(var[i])])) for i in range(num_row)]).reshape((num_row, 1))
                vars_h5[var_names_h5[ip]] = var

            vars = {}
            for var_name, var_name_h5 in var_name_dict.items():
                try:
                    vars[var_name] = vars_h5[var_name_h5]
                except KeyError:
                    vars[var_name] = None

            num_row = inds_ran_min.shape[0]
            # unix time to datetime
            utime1 = vars_h5['UT1_UNIX'].flatten()
            dt1 = dttool.convert_unix_time_to_datetime_cftime(utime1)
            var = dt1.reshape(num_row, 1)
            var_name = 'DATETIME_1'
            vars[var_name] = var

            utime2 = vars_h5['UT2_UNIX'].flatten()
            dt2 = dttool.convert_unix_time_to_datetime_cftime(utime2)
            var = dt2.reshape(num_row, 1)
            var_name = 'DATETIME_2'
            vars[var_name] = var

            vars['DATETIME'] = vars['DATETIME_1'] + (vars['DATETIME_2'] - vars['DATETIME_1']) / 2
            vars['T_e'] = vars['T_i'] * vars['T_r']
            vars['T_e_err'] = vars['T_e'] * np.sqrt((vars['T_i_err'] / vars['T_i']) ** 2
                                                    + (vars['T_r_err'] / vars['T_r']) ** 2)
            vars['AZ'] = vars['AZ'] % 360.
            inds = np.where(np.isnan(vars['HEIGHT']))
            for i in range(len(inds[0])):
                ind_0 = inds[0][i]
                ind_1 = inds[1][i]
                x0 = np.arange(vars['HEIGHT'].shape[0])
                y0 = vars['HEIGHT'][:, ind_1]
                xp = x0[np.where(np.isfinite(y0))[0]]
                yp = y0[np.isfinite(y0)]
                vars['HEIGHT'][ind_0, ind_1] = np.interp(x0[ind_0], xp, yp)

                x0 = np.arange(vars['RANGE'].shape[0])
                y0 = vars['RANGE'][:, ind_1]
                xp = x0[np.isfinite(y0)]
                yp = y0[np.isfinite(y0)]
                vars['RANGE'][ind_0, ind_1] = np.interp(x0[ind_0], xp, yp)

            self.variables = vars
            self.metadata = metadata
        # raise NotImplemented



class Loader(LoaderModel):
    """
    Load SWARM 2Hz or 16HZ TII data products. Currently support versions higher than "0301".

    The class is a hierarchy of :class:`SWARM data LoaderModel <geospacelab.datahub.sources.esa_eo.swarm.loader.LoaderModel>`

    """
    def __init__(self, *args, **kwargs):
        kwargs.setdefault('variable_name_dict', default_variable_name_dict)
        super(Loader, self).__init__(*args, **kwargs)