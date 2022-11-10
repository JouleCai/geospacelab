import numpy as np
from scipy.signal import argrelextrema
import datetime
from scipy.interpolate import griddata
from scipy.signal import butter, lfilter, freqz
import scipy.signal as sig

from geospacelab.datahub import DatasetUser
from geospacelab.toolbox.utilities import pydatetime as dttool


class LEOToolbox(DatasetUser):

    def __init__(self, dt_fr=None, dt_to=None):
        super().__init__(dt_fr, dt_to)

        self.add_variable(var_name='SC_GEO_LAT')
        self.add_variable(var_name='SC_GEO_LON')
        self.add_variable(var_name='SC_GEO_ALT')
        self.add_variable(var_name='SC_GEO_LST')
        self.add_variable(var_name='SC_DATETIME')
        self.add_variable(var_name='SC_AACGM_LAT')
        self.add_variable(var_name='SC_AACGM_LON')
        self.add_variable(var_name='SC_AACGM_MLT')
        self.add_variable(var_name='SC_APEX_LAT')
        self.add_variable(var_name='SC_APEX_LON')
        self.add_variable(var_name='SC_APEX_MLT')
        self.ascending_nodes = {}
        self.descending_nodes = {}
        self.sector_cs = 'GEO'
        self.sectors = {}

    def search_orbit_nodes(self):
        glat = self['SC_GEO_LAT'].value.flatten()
        glon = self['SC_GEO_LON'].value.flatten()
        lst = self['SC_GEO_LST'].value.flatten()
        dts = self['SC_DATETIME'].value.flatten()
        ind_1 = argrelextrema(np.abs(glat), np.less)[0]
        ind_2 = argrelextrema(glat, np.greater)[0]
        inds_asc = []
        inds_dsc = []
        for ind_N in ind_2:
            if ind_N < ind_1[0]:
                inds_dsc.append(ind_1[0])
            elif ind_N > ind_1[-1]:
                inds_asc.append(ind_1[-1])
            else:
                ind_between = np.where(ind_1 > ind_N)[0]
                inds_asc.append(ind_1[ind_between[0]-1])
                inds_dsc.append(ind_1[ind_between[0]])
        self.ascending_nodes['INDEX'] = inds_asc
        self.ascending_nodes['GEO_LON'] = glon[inds_asc]
        self.ascending_nodes['GEO_LST'] = lst[inds_asc]
        self.ascending_nodes['DATETIME'] = dts[inds_asc]
        self.descending_nodes['INDEX']= inds_dsc
        self.descending_nodes['GEO_LON'] = glon[inds_dsc]
        self.descending_nodes['GEO_LST'] = lst[inds_dsc]
        self.descending_nodes['DATETIME'] = dts[inds_asc]

    def group_by_sector(self, sector_name, boundary_lat, sector_cs='GEO'):
        self.sector_cs = sector_cs
        if not dict(self.ascending_nodes):
            self.search_orbit_nodes()

        glat = self['SC_GEO_LAT'].value.flatten()
        glon = self['SC_GEO_LON'].value.flatten()
        lst = self['SC_GEO_LST'].value.flatten()
        
        if self.sector_cs == 'AACGM':
            mlat = self['SC_AACGM_LAT'].value.flatten()
            mlon = self['SC_AACGM_LON'].value.flatten()
            mlt = self['SC_AACGM_MLT'].value.flatten()
        elif self.sector_cs == 'APEX':
            mlat = self['SC_APEX_LAT'].value.flatten()
            mlon = self['SC_APEX_LON'].value.flatten()
            mlt = self['SC_APEX_MLT'].value.flatten()
        
        dts = self['SC_DATETIME'].value.flatten()
        
        inds_asc = self.ascending_nodes['INDEX']
        inds_dsc = self.descending_nodes['INDEX']

        if self.sector_cs == 'GEO':
            lat = glat
            lt = lst
        elif self.sector_cs == 'AACGM':
            lat = mlat
            lt = mlt
        elif self.sector_cs == 'APEX':
            lat = mlat
            lt = mlt
        else:
            raise NotImplementedError

        inds_center_N = argrelextrema(lat, np.greater)[0]
        # inds_center_N = inds_center_N[np.where(lat[inds_center_N] > boundary_lat)[0]]

        inds_center_S = argrelextrema(lat, np.less)[0]
        # inds_center_S = inds_center_S[np.where(lat[inds_center_S] < -np.abs(boundary_lat))[0]]

        # Sector centered at the northern pole from the ascending node towards the descending node.
        if sector_name == 'N':
            sector = np.zeros((dts.size,))
            dts_c = np.zeros_like(dts)

            pseudo_lat = np.empty_like(sector)
            pseudo_lat[:] = np.nan
            for num, ind_rec in enumerate(inds_center_N):

                ind_tmp = np.where(inds_center_S < ind_rec)[0]
                if list(ind_tmp):
                    ind_rec_S1 = inds_center_S[ind_tmp[-1]]
                else:
                    ind_rec_S1 = 0
                ind_tmp = np.where(inds_center_S > ind_rec)[0]
                if list(ind_tmp):
                    ind_rec_S2 = inds_center_S[ind_tmp[0]]
                else:
                    ind_rec_S2 = len(lat) - 1
                inds_seg = np.array(range(ind_rec_S1, ind_rec_S2))

                dt_c = dts[ind_rec]
                lat_seg = lat[inds_seg]
                dts_seg = dts[inds_seg]
                delta_t_seg = np.array([(dt1 - dt_c).total_seconds() / 60. for dt1 in dts_seg])
                inds_seg_seg = np.where((lat_seg > boundary_lat) & (np.abs(delta_t_seg) < 60.))[0]

                inds_sector = inds_seg[inds_seg_seg]
                sector[inds_sector] = num + 1
                dts_c[inds_sector] = dt_c

                lat_seg = np.where(dts_seg < dt_c, lat_seg, 180. - lat_seg)
                pseudo_lat[inds_sector] = lat_seg[inds_seg_seg]

            self.add_variable(var_name='SECTOR_N', value=sector.reshape((sector.size, 1)))
            self.add_variable(var_name='SECTOR_N_DATETIME', value=dts_c.reshape((sector.size, 1)))
            self.add_variable(var_name='SECTOR_N_PSEUDO_LAT', value=pseudo_lat.reshape((sector.size, 1)))
            self.sectors['N'] = {
                'BOUNDARY_LAT': boundary_lat,
                'PSEUDO_LAT_RANGE': [boundary_lat, 180. - boundary_lat],
                'VARIABLE_NAMES': ['SECTOR_N', 'SECTOR_N_DATETIME', 'SECTOR_N_PSEUDO_LAT']
            }
        
        # Sector centered at the southern pole from the descending node towards the ascending node.
        elif sector_name == 'S':
            sector = np.zeros((dts.size,))
            dts_c = np.zeros_like(dts)

            pseudo_lat = np.empty_like(sector)
            pseudo_lat[:] = np.nan
            for num, ind_rec in enumerate(inds_center_S):

                ind_tmp = np.where(inds_center_N < ind_rec)[0]
                if list(ind_tmp):
                    ind_rec_N1 = inds_center_N[ind_tmp[-1]]
                else:
                    ind_rec_N1 = 0
                ind_tmp = np.where(inds_center_N > ind_rec)[0]
                if list(ind_tmp):
                    ind_rec_N2 = inds_center_N[ind_tmp[0]]
                else:
                    ind_rec_N2 = len(lat) - 1
                inds_seg = np.array(range(ind_rec_N1, ind_rec_N2))

                dt_c = dts[ind_rec]
                lat_seg = lat[inds_seg]
                dts_seg = dts[inds_seg]
                delta_t_seg = np.array([(dt1 - dt_c).total_seconds() / 60. for dt1 in dts_seg])
                inds_seg_seg = np.where((lat_seg < -np.abs(boundary_lat)) & (np.abs(delta_t_seg) < 60.))[0]

                inds_sector = inds_seg[inds_seg_seg]
                sector[inds_sector] = num + 1
                dts_c[inds_sector] = dt_c

                lat_seg = np.where(dts_seg < dt_c, 180. - lat_seg, 360. + lat_seg)
                pseudo_lat[inds_sector] = lat_seg[inds_seg_seg]

            self.add_variable(var_name='SECTOR_S', value=sector.reshape((sector.size, 1)))
            self.add_variable(var_name='SECTOR_S_DATETIME', value=dts_c.reshape((sector.size, 1)))
            self.add_variable(var_name='SECTOR_S_PSEUDO_LAT', value=pseudo_lat.reshape((sector.size, 1)))
            self.sectors['S'] = {
                'BOUNDARY_LAT': boundary_lat,
                'PSEUDO_LAT_RANGE': [180. + np.abs(boundary_lat), 360. - np.abs(boundary_lat)],
                'VARIABLE_NAMES': ['SECTOR_S', 'SECTOR_S_DATETIME', 'SECTOR_S_PSEUDO_LAT']
            }

        # Sector centered at the asending node from south towards north.
        elif sector_name == 'ASC':
            sector = np.zeros((dts.size,))
            dts_c = np.zeros_like(dts)

            pseudo_lat = np.empty_like(sector)
            pseudo_lat[:] = np.nan
            for num, ind_rec in enumerate(inds_asc):

                ind_tmp = np.where(inds_center_S < ind_rec)[0]
                if list(ind_tmp):
                    ind_rec_S1 = inds_center_S[ind_tmp[-1]]
                else:
                    ind_rec_S1 = 0
                ind_tmp = np.where(inds_center_N > ind_rec)[0]
                if list(ind_tmp):
                    ind_rec_N1 = inds_center_N[ind_tmp[0]]
                else:
                    ind_rec_N1 = len(lat) - 1 
                inds_seg = np.array(range(ind_rec_S1, ind_rec_N1))

                dt_c = dts[ind_rec]
                lat_seg = lat[inds_seg]
                dts_seg = dts[inds_seg]
                delta_t_seg = np.array([(dt1 - dt_c).total_seconds() / 60. for dt1 in dts_seg])
                inds_seg_seg = np.where((np.abs(lat_seg) < np.abs(boundary_lat)) & (np.abs(delta_t_seg) < 60.))[0]

                inds_sector = inds_seg[inds_seg_seg]
                sector[inds_sector] = num + 1
                dts_c[inds_sector] = dt_c

                lat_seg = np.where(lat_seg < 0, lat_seg, lat_seg)
                pseudo_lat[inds_sector] = lat_seg[inds_seg_seg]

            self.add_variable(var_name='SECTOR_ASC', value=sector.reshape((sector.size, 1)))
            self.add_variable(var_name='SECTOR_ASC_DATETIME', value=dts_c.reshape((sector.size, 1)))
            self.add_variable(var_name='SECTOR_ASC_PSEUDO_LAT', value=pseudo_lat.reshape((sector.size, 1)))
            self.sectors['ASC'] = {
                'BOUNDARY_LAT': boundary_lat,
                'PSEUDO_LAT_RANGE': [-np.abs(boundary_lat), np.abs(boundary_lat)],
                'VARIABLE_NAMES': ['SECTOR_ASC', 'SECTOR_ASC_DATETIME', 'SECTOR_ASC_PSEUDO_LAT']
            }
            
            # Sector centered at the asending node from south towards north.
        elif sector_name == 'DSC':
            sector = np.zeros((dts.size,))
            dts_c = np.zeros_like(dts)

            pseudo_lat = np.empty_like(sector)
            pseudo_lat[:] = np.nan
            for num, ind_rec in enumerate(inds_dsc):

                ind_tmp = np.where(inds_center_N < ind_rec)[0]
                if list(ind_tmp):
                    ind_rec_N1 = inds_center_N[ind_tmp[-1]]
                else:
                    ind_rec_N1 = 0
                ind_tmp = np.where(inds_center_S > ind_rec)[0]
                if list(ind_tmp):
                    ind_rec_S1 = inds_center_S[ind_tmp[0]]
                else:
                    ind_rec_S1 = len(lat) - 1
                inds_seg = np.array(range(ind_rec_N1, ind_rec_S1))

                dt_c = dts[ind_rec]
                lat_seg = lat[inds_seg]
                dts_seg = dts[inds_seg]
                delta_t_seg = np.array([(dt1 - dt_c).total_seconds() / 60. for dt1 in dts_seg])
                inds_seg_seg = np.where((np.abs(lat_seg) < np.abs(boundary_lat)) & (np.abs(delta_t_seg) < 60.))[0]

                inds_sector = inds_seg[inds_seg_seg]
                sector[inds_sector] = num + 1
                dts_c[inds_sector] = dt_c

                lat_seg = np.where(lat_seg > 0, 180. - lat_seg, 180. - lat_seg)
                pseudo_lat[inds_sector] = lat_seg[inds_seg_seg]

            self.add_variable(var_name='SECTOR_DSC', value=sector.reshape((sector.size, 1)))
            self.add_variable(var_name='SECTOR_DSC_DATETIME', value=dts_c.reshape((sector.size, 1)))
            self.add_variable(var_name='SECTOR_DSC_PSEUDO_LAT', value=pseudo_lat.reshape((sector.size, 1)))
            self.sectors['DSC'] = {
                'BOUNDARY_LAT': boundary_lat,
                'PSEUDO_LAT_RANGE': [180. - np.abs(boundary_lat), 180. + np.abs(boundary_lat)],
                'VARIABLE_NAMES': ['SECTOR_DSC', 'SECTOR_DSC_DATETIME', 'SECTOR_DSC_PSEUDO_LAT']
            }
        else:
            raise NotImplementedError
                
    def griddata_by_sector(self, sector_name=None, variable_names=None, x_grid_res=20*60, y_grid_res=0.5):
        self.visual = 'on'
        dts_c = self['_'.join(('SECTOR', sector_name, 'DATETIME'))].value.flatten()
        dts = self['SC_DATETIME'].value.flatten()
        sector = self['SECTOR_' + sector_name].value.flatten()
        lat = self['_'.join(('SECTOR', sector_name, 'PSEUDO_LAT'))].value.flatten()
        boundary_lat = self.sectors[sector_name]['BOUNDARY_LAT']
        lat_range = self.sectors[sector_name]['PSEUDO_LAT_RANGE']

        x_data = dts_c[sector > 0]
        y_data = lat[sector > 0]
        is_finite = np.isfinite(y_data)
        dt0 = dttool.get_start_of_the_day(self.dt_fr)
        sectime = np.array([(dt - dt0).total_seconds() for dt in x_data])
        x = sectime
        y = y_data
        # max_x = np.ceil(np.max(x) / self.xgrid_res) * self.xgrid_res
        # min_x = np.floor(np.min(x) / self.xgrid_res) * self.xgrid_res
        total_seconds = (dts[-1] - dt0).total_seconds()
        min_x = 0
        max_x = total_seconds

        nx = int(np.ceil((max_x - min_x) / x_grid_res) + 1)
        ny = int(np.ceil(np.diff(lat_range) / y_grid_res) + 1)

        grid_x, grid_y = np.meshgrid(np.linspace(min_x, max_x, nx),
                                    np.linspace(lat_range[0], lat_range[1], ny))

        grid_lat = griddata((x, y), y, (grid_x, grid_y), method='nearest')
        if self.sector_cs == 'AACGM':
            mask = np.abs(grid_y - grid_lat) > 2
            which_lat = 'AACGM_LAT'
        if self.sector_cs == 'APEX':
            mask = np.abs(grid_y - grid_lat) > 2
            which_lat = 'APEX_LAT' 
        else:
            mask = np.abs(grid_y - grid_lat) > 1
            which_lat = 'GEO_LAT'

        method = 'linear'
        for var_name in variable_names:
            vrb = self[var_name].value.flatten()[sector>0]
            grid_z = griddata((x, y), vrb, (grid_x, grid_y), method=method)
            grid_z[mask] = np.nan
            var_name_in = '_'.join(('SECTOR', sector_name, 'GRID', var_name))
            self.sectors[sector_name]['VARIABLE_NAMES'].append(var_name_in) 
            
            # self.add_variable(var_name=var_name_in, value=grid_z.T)
            self[var_name_in] = self[var_name].clone(omit_attrs='visual')
            self[var_name_in].visual = 'new'
            var = self[var_name_in]
            var.value = grid_z.T
            var.set_depend(0, {'UT': '_'.join(('SECTOR', sector_name, 'GRID_DATETIME'))})
            var.set_depend(1, {which_lat: '_'.join(('SECTOR', sector_name, 'GRID_LAT'))})
            var.visual.axis[0].data = '@d.' + '_'.join(('SECTOR', sector_name, 'GRID_DATETIME'))
            var.visual.axis[1].data = '@d.' + '_'.join(('SECTOR', sector_name, 'GRID_LAT'))
            var.visual.axis[1].label = 'GLAT' if self.sector_cs == 'GEO' else 'MLAT'
            var.visual.axis[1].unit = r'$^\circ$'
            var.visual.axis[1].lim = lat_range
            var.visual.axis[2].data = '@v.value'
            var.visual.axis[2].data_scale = self[var_name].visual.axis[1].data_scale
            var.visual.axis[2].label = '@v.label'
            var.visual.axis[2].unit = '@v.unit'
            var.visual.plot_config.style = '2P'
            var.visual.plot_config.pcolormesh.update(cmap='jet')

        grid_x_name = '_'.join(('SECTOR', sector_name, 'GRID', 'X'))
        grid_x = grid_x.T
        self.add_variable(var_name=grid_x_name, value=grid_x)
        grid_y_name = '_'.join(('SECTOR', sector_name, 'GRID', 'Y'))
        grid_y = grid_y.T
        self.add_variable(var_name=grid_y_name, value=grid_y)
        var_name = '_'.join(('SECTOR', sector_name, 'GRID_DATETIME'))
        grid_dts = np.array([dt0 + datetime.timedelta(seconds=sec) for sec in grid_x[:, 0]])
        self.add_variable(var_name=var_name, value=grid_dts[:, np.newaxis])
        var_name = '_'.join(('SECTOR', sector_name, 'GRID_LAT'))
        self.add_variable(var_name=var_name, value=grid_y[0, :][np.newaxis, :])

    def filter_by_time(self, variable_names=None, time_window=40., time_res=None):


        return

    @staticmethod
    def format_pseudo_lat_label(ax, sector_name, is_integer=True):
        if is_integer:
            lat_format = '{:4.0f}'
        else:
            lat_format = '{:5.1f}'
        y_lim = ax.get_ylim()
        y_max = np.max(y_lim)
        y_min = np.min(y_lim)
        yticks = ax.get_yticks()
        ylabels = []
        n1 = 0
        n2 = len(yticks) - 1
        n3 = 0
        for ind, pseudo_lat in enumerate(yticks):
            if pseudo_lat < y_min:
                n1 = n1 + 1
                ylabels.append('')
                continue
            if pseudo_lat > y_max:
                n2 = n2 - 1
                ylabels.append('')
                continue
            if sector_name == 'N':
                lat = pseudo_lat if pseudo_lat <= 90. else 180. - pseudo_lat
                if lat == 90.:
                    n3 = ind

            elif sector_name == 'S':
                lat = 180. - pseudo_lat if pseudo_lat <= 270. else pseudo_lat - 360.
                if lat == -90:
                    n3 = ind

            elif sector_name == 'ASC':
                lat = pseudo_lat
                if lat == 0.:
                    n3 = ind

            elif sector_name == 'DSC':
                lat = 180. - pseudo_lat
                if lat == 0.:
                    n3 = ind

            else:
                raise NotImplementedError
            if lat > 0:
                lat_format_1 = r'' + lat_format + r'$^\circ$N'
                lb = lat_format_1.format(lat)
            elif lat < 0:
                lat_format_1 = r'' + lat_format + r'$^\circ$S'
                lb = lat_format_1.format(np.abs(lat))
            else:
                lat_format_1 = r'' + '{:4.0f}' + r'$^\circ$'
                lb = lat_format_1.format(lat) 
            ylabels.append(lb)

        if sector_name == 'N':
            ylabels[n1] = 'ASC:' + ylabels[n1]
            ylabels[n2] = 'DSC:' + ylabels[n2]
            ylabels[n3] = 'Pole->' + ylabels[n3]
        elif sector_name == 'S':
            ylabels[n1] = 'DSC:' + ylabels[n1]
            ylabels[n2] = 'ASC:' + ylabels[n2]
            ylabels[n3] = 'Pole->' + ylabels[n3]
        elif sector_name == 'ASC':
            ylabels[n1] = '' + ylabels[n1]
            ylabels[n2] = '' + ylabels[n2]
            ylabels[n3] = 'ASC->' + ylabels[n3]
        elif sector_name == 'DSC':
            ylabels[n1] = '' + ylabels[n1]
            ylabels[n2] = '' + ylabels[n2]
            ylabels[n3] = 'DSC->' + ylabels[n3]
        else:
            raise NotImplementedError
        ax.set_yticks(yticks)
        ax.set_yticklabels(ylabels)
        ax.set_ylim(y_lim)

        return

    def smooth_along_track(self, time_window=450, time_res=None, variable_names=None, method='savgol', residuals='on'):
        self.visual = 'on'
        dts = self['SC_DATETIME'].value.flatten()
        dt0 = dttool.get_start_of_the_day(self.dt_fr)
        sectime = np.array([(dt - dt0).total_seconds() for dt in dts])
        if time_res is None:
            diff_secs = np.diff(sectime)
            time_res = np.median(diff_secs)

        N = int(np.fix(time_window/time_res))

        for var_name in variable_names:
            vrb = self[var_name].value.flatten()
            vrb_avg = self.smooth_savgol(dts, vrb, N, time_res)
            var_name_in = var_name + '_AVG'
            self[var_name_in] = self[var_name].clone()
            self[var_name_in].value = vrb_avg[:, np.newaxis]
            self[var_name_in].label = 'smoothed ' + self[var_name_in].label
            if residuals:
                var_name_in = var_name + '_RSD'
                self[var_name_in] = self[var_name].clone()
                self[var_name_in].value = self[var_name].value - self[var_name + '_AVG'].value
                self[var_name_in].label = 'RSD'
                self[var_name_in].visual.axis[1].lim = [-np.inf, np.inf]
                var_name_in = var_name + '_RSD_PERC'
                self[var_name_in] = self[var_name].clone()
                self[var_name_in].value = (self[var_name].value - self[var_name + '_AVG'].value) / \
                    self[var_name + '_AVG'].value
                self[var_name_in].label = "RSD pct."
                self[var_name_in].unit = '%'
                self[var_name_in].unit_label = None
                self[var_name_in].visual.axis[1].data_scale = 100.

    @staticmethod
    def smooth_savgol(dts, data, box_pts, t_res):
        delta_sectime = 1
        sectime, _ = dttool.convert_datetime_to_sectime(dts)
        ind_nan = np.isnan(data)
        sectime_i = np.arange(0, sectime[-1], delta_sectime)
        data_i = np.interp(sectime_i, sectime[~ind_nan], data[~ind_nan])

        order = 1
        window_size = int(np.floor(t_res / delta_sectime * box_pts / 2) * 2 + 1)
        y = sig.savgol_filter(data_i, window_size, order)
        data_new = np.interp(sectime, sectime_i, y)
        data_new[ind_nan] = np.nan
        return data_new