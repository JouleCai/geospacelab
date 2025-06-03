import numpy as np
from scipy.signal import argrelextrema
import datetime
from scipy.interpolate import griddata, interp1d
from scipy.signal import butter, lfilter, freqz
import scipy.signal as sig

from geospacelab.datahub import DatasetUser
from geospacelab.toolbox.utilities import pydatetime as dttool
import geospacelab.toolbox.utilities.numpymath as npmath
from geospacelab.cs import GEOCSpherical


class LEOToolbox(DatasetUser):

    def __init__(self, dt_fr=None, dt_to=None, visual='on'):
        super().__init__(dt_fr, dt_to, visual=visual)

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
        self.northern_nodes = {}
        self.southern_nodes = {}
        self.sector_cs = 'GEO'
        self.sectors = {}

    def search_orbit_nodes(self, data_interval=1, t_res=None):
        def check_nodes(nodes):
            dts_nodes = nodes['DATETIME']
            sectimes_nodes, _ = dttool.convert_datetime_to_sectime(dts_nodes)
            diff_sectimes = np.diff(sectimes_nodes)
            if t_res is not None:
                sectime_res_1 = t_res
            else:
                sectime_res_1 = np.median(np.diff(sectimes_nodes))
            inds_abnormal = np.where(diff_sectimes<sectime_res_1*0.6)[0]
            if list(inds_abnormal):
                iiii = np.array(inds_abnormal) + 1
                nodes['INDEX'] = np.delete(nodes['INDEX'], iiii)
                nodes['GEO_LAT'] = np.delete(nodes['GEO_LAT'], iiii)
                nodes['GEO_LON'] = np.delete(nodes['GEO_LON'], iiii)
                nodes['GEO_ALT'] = np.delete(nodes['GEO_ALT'], iiii)
                nodes['GEO_LST'] = np.delete(nodes['GEO_LST'], iiii)
                nodes['DATETIME'] = np.delete(nodes['DATETIME'], iiii)


        glat = self['SC_GEO_LAT'].value.flatten()
        glat_ = glat[0::data_interval]
        glon = self['SC_GEO_LON'].value.flatten()
        alt = self['SC_GEO_ALT'].flatten()
        lst = self['SC_GEO_LST'].value.flatten()
        dts = self['SC_DATETIME'].value.flatten()
        ind_1 = argrelextrema(np.abs(glat_), np.less, )[0]
        ind_1 = ind_1 * data_interval
        ind_2 = argrelextrema(glat_, np.greater, )[0]
        ind_2 = ind_2 * data_interval
        ind_3 = argrelextrema(glat_, np.less)[0]
        ind_3 = ind_3 * data_interval
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
        self.ascending_nodes['INDEX'] = np.array(inds_asc)
        self.ascending_nodes['GEO_LON'] = glon[inds_asc] 
        self.ascending_nodes['GEO_LAT'] = glat[inds_asc] 
        self.ascending_nodes['GEO_ALT'] = alt[inds_asc]  
        self.ascending_nodes['GEO_LST'] = lst[inds_asc]
        self.ascending_nodes['DATETIME'] = dts[inds_asc]
        check_nodes(self.ascending_nodes)

        self.descending_nodes['INDEX']= np.array(inds_dsc)
        self.descending_nodes['GEO_LON'] = glon[inds_dsc]
        self.descending_nodes['GEO_LAT'] = glat[inds_dsc]
        self.descending_nodes['GEO_ALT'] = alt[inds_dsc]
        self.descending_nodes['GEO_LST'] = lst[inds_dsc]
        self.descending_nodes['DATETIME'] = dts[inds_dsc]
        check_nodes(self.descending_nodes)

        self.northern_nodes['INDEX'] = ind_2
        self.northern_nodes['GEO_LON'] = glon[ind_2]
        self.northern_nodes['GEO_LAT'] = glat[ind_2]
        self.northern_nodes['GEO_ALT'] = alt[ind_2]
        self.northern_nodes['GEO_LST'] = lst[ind_2]
        self.northern_nodes['DATETIME'] = dts[ind_2]
        check_nodes(self.northern_nodes)

        self.southern_nodes['INDEX'] = ind_3
        self.southern_nodes['GEO_LON'] = glon[ind_3]
        self.southern_nodes['GEO_LAT'] = glat[ind_3]
        self.southern_nodes['GEO_ALT'] = alt[ind_3]
        self.southern_nodes['GEO_LST'] = lst[ind_3]
        self.southern_nodes['DATETIME'] = dts[ind_3]
        check_nodes(self.southern_nodes)

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

        # inds_center_N = argrelextrema(lat, np.greater)[0]
        inds_center_N = self.northern_nodes['INDEX']
        iii = np.where(lat[inds_center_N] > (np.max(lat[inds_center_N]))/1.5)[0]
        inds_center_N = inds_center_N[iii]
        # inds_center_N = inds_center_N[np.where(lat[inds_center_N] > boundary_lat)[0]]

        # inds_center_S = argrelextrema(lat, np.less)[0]
        inds_center_S = self.southern_nodes['INDEX']
        iii = np.where(lat[inds_center_S] < (np.min(lat[inds_center_S]))/1.5)[0]
        inds_center_S = inds_center_S[iii]
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
                
    def griddata_by_sector(
            self, sector_name=None, variable_names=None, x_grid_res=20*60, y_grid_res=0.5, along_track_interp=True,
            x_data_res=None, y_data_res=None, along_track_binning=False, 
            ):
        self.visual = 'on'
        dts_c = self['_'.join(('SECTOR', sector_name, 'DATETIME'))].value.flatten()
        dts = self['SC_DATETIME'].value.flatten()
        sector = self['SECTOR_' + sector_name].value.flatten()
        lat = self['_'.join(('SECTOR', sector_name, 'PSEUDO_LAT'))].value.flatten()
        boundary_lat = self.sectors[sector_name]['BOUNDARY_LAT']
        lat_range = self.sectors[sector_name]['PSEUDO_LAT_RANGE']

        if along_track_interp:
            x_data = dts[sector > 0]
        else:
            x_data = dts_c[sector > 0]
        
        x_data_1 = dts_c[sector>0]
        y_data = lat[sector > 0]
        is_finite = np.isfinite(y_data)
        dt0 = dttool.get_start_of_the_day(self.dt_fr)
        sectime = np.array([(dt - dt0).total_seconds() for dt in x_data])
        x = sectime
        y = y_data
        
        x_1 = np.array([(dt - dt0).total_seconds() for dt in x_data_1]) 
        # max_x = np.ceil(np.max(x) / self.xgrid_res) * self.xgrid_res
        # min_x = np.floor(np.min(x) / self.xgrid_res) * self.xgrid_res

        x_unique = np.unique(dts_c[sector > 0])
        x_unique = [(t - dt0).total_seconds() for t in x_unique]
        if x_data_res is None:
            x_data_res = np.median(np.diff(x_unique))

        if y_data_res is None:
            y_data_res = np.median(np.diff(y_data[y_data < np.abs(np.diff(lat_range))]/2))
            if y_grid_res > y_data_res:
                y_data_res = y_grid_res

        ny = int(np.ceil(np.diff(lat_range) / y_grid_res) + 1)
        if x_grid_res is None:
            grid_x, grid_y = np.meshgrid(x_unique, np.linspace(lat_range[0], lat_range[1], ny))
            x_interp = False
            x_grid_res=x_data_res
        else:
            min_x = np.floor((self.dt_fr - dt0).total_seconds() / x_grid_res) * x_grid_res
            max_x = np.ceil((self.dt_to - dt0).total_seconds() / x_grid_res) * x_grid_res
            # total_seconds = (dts[-1] - dt0).total_seconds()
            # min_x = 0
            # max_x = total_seconds
            # nx = int(np.ceil((max_x - min_x) / x_grid_res))
            # grid_x, grid_y = np.meshgrid(np.linspace(min_x, max_x, nx),
            #                            np.linspace(lat_range[0], lat_range[1], ny))
            grid_x, grid_y = np.meshgrid(
                np.arange(min_x, max_x, x_grid_res),
                np.linspace(lat_range[0], lat_range[1], ny)
            )
            x_interp = True

        grid_lat = griddata((x_1, y), y, (grid_x, grid_y), method='nearest')

        if self.sector_cs == 'AACGM':
            mask_y = np.abs(grid_y - grid_lat) > y_data_res * 2
            which_lat = 'AACGM_LAT'
        if self.sector_cs == 'APEX':
            mask_y = np.abs(grid_y - grid_lat) > y_data_res * 2
            which_lat = 'APEX_LAT' 
        else:
            mask_y = np.abs(grid_y - grid_lat) > y_data_res * 1.2
            which_lat = 'GEO_LAT'

        grid_sectime = griddata((x_1, y), x_1, (grid_x, grid_y), method='nearest')
        mask_x = np.abs(grid_x - grid_sectime) > x_data_res * 1.2

        # remove gaps
        i_mask_x = np.where(mask_x)
        rec_bounds_ind = {}
        if list(i_mask_x[0]):
            for ii, jj in zip(*i_mask_x):
                xx = grid_x[ii, jj]
                x_ii = grid_x[ii, :].flatten()
                if xx in rec_bounds_ind.keys():
                    bound_x_1 = rec_bounds_ind[xx][0]
                    bound_x_2 = rec_bounds_ind[xx][1]
                else:
                    iii = np.where(x_unique<xx)[0]
                    if not list(iii):
                        bound_x_1 = x_ii[0] - x_data_res
                    else:
                        bound_x_1 = x_unique[iii[-1]] - x_grid_res
                    iii = np.where(x_unique > xx)[0]
                    if not list(iii):
                        bound_x_2 = x_ii[-1] + x_grid_res
                    else:
                        bound_x_2 = x_unique[iii[0]] + x_data_res
                    rec_bounds_ind[xx] = [bound_x_1, bound_x_2]
                mask_x[ii, (x_ii >= bound_x_1) & (x_ii <= bound_x_2)] = True
            pass

        method = 'linear'
        for var_name in variable_names:
            # print(var_name)
            vrb = self[var_name].value.flatten()[sector>0]
            if along_track_binning:
                sector_1 = sector[sector>0]
                n_tracks = int(np.max(sector))
                n_lats = ny
                xx_1 = np.empty((n_tracks, n_lats))
                zz_1 = np.empty((n_tracks, n_lats))
                yy_1 = grid_y[:, 0].flatten()
                grid_z = np.ones_like(grid_x)*np.nan
                for ii in np.arange(1, n_tracks+1):
                    xd = x[sector_1 == ii]
                    yd = y[sector_1 == ii]
                    zd = vrb[sector_1 == ii]
                    if (not list(xd)) or (not list(yd)):
                        continue
                        
                    f = interp1d(yd, xd, bounds_error=False, fill_value= 'extrapolate')
                    x_i = f(yy_1)
                    xx_1[ii-1, :] = x_i

                    if 'LON' in var_name:
                        z_i = npmath.interp_period_data(yd, zd, yy_1,  period=360., method='linear', bounds_error=False)
                    elif 'LST' in var_name:
                        z_i = npmath.interp_period_data(yd, zd, yy_1,  period=24., method='linear', bounds_error=False)
                    else:
                        # f = interp1d(yd, zd, bounds_error=False, fill_value='extrapolate')
                        # z_i = f(yy_1)
                        z_i = np.ones_like(yy_1) * np.nan       
                        for j, yyy in enumerate(yy_1):
                            inds_y = np.where((yd >= yyy-y_grid_res/2) & (yd < yyy+y_grid_res/2))[0]
                            if list(inds_y):
                                z_i[j] = np.nanmean(zd[inds_y])
                    zz_1[ii-1, :] = z_i 
                for ii in range(grid_x.shape[0]):
                    xd = xx_1[:, ii].flatten()
                    zd = zz_1[:, ii].flatten()
                    if x_interp:
                        if 'LON' in var_name:
                            z_i = npmath.interp_period_data(xd, zd, grid_x[0],  period=360., method='linear', bounds_error=False)
                        elif 'LST' in var_name:
                            z_i = npmath.interp_period_data(xd, zd, grid_x[0],  period=24., method='linear', bounds_error=False)
                        else:
                            f = interp1d(xd, zd, bounds_error=False, fill_value='extrapolate')
                            z_i = f(grid_x[0])
                        grid_z[ii, :] = z_i
                    else:
                        grid_z[ii, :] = zd 
            elif along_track_interp:
                sector_1 = sector[sector>0]
                n_tracks = int(np.max(sector))
                n_lats = ny
                xx_1 = np.empty((n_tracks, n_lats))
                zz_1 = np.empty((n_tracks, n_lats))
                yy_1 = grid_y[:, 0].flatten()
                grid_z = np.ones_like(grid_x) * np.nan
                for ii in np.arange(1, n_tracks+1):
                    xd = x[sector_1 == ii]
                    yd = y[sector_1 == ii]
                    zd = vrb[sector_1 == ii]
                    
                    if (not list(xd)) or (not list(yd)):
                        continue

                    f = interp1d(yd, xd, bounds_error=False, fill_value= 'extrapolate')
                    x_i = f(yy_1)
                    xx_1[ii-1, :] = x_i

                    if 'LON' in var_name:
                        z_i = npmath.interp_period_data(yd, zd, yy_1,  period=360., method='linear', bounds_error=False)
                    elif 'LST' in var_name:
                        z_i = npmath.interp_period_data(yd, zd, yy_1,  period=24., method='linear', bounds_error=False)
                    else:
                        f = interp1d(yd, zd, bounds_error=False, fill_value='extrapolate')
                        z_i = f(yy_1)
                    zz_1[ii-1, :] = z_i 
                for ii in range(grid_x.shape[0]):
                    xd = xx_1[:, ii].flatten()
                    zd = zz_1[:, ii].flatten()
                    if x_interp:
                        if 'LON' in var_name:
                            z_i = npmath.interp_period_data(xd, zd, grid_x[0],  period=360., method='linear', bounds_error=False)
                        elif 'LST' in var_name:
                            z_i = npmath.interp_period_data(xd, zd, grid_x[0],  period=24., method='linear', bounds_error=False)
                        else:
                            f = interp1d(xd, zd, bounds_error=False, fill_value='extrapolate')
                            z_i = f(grid_x[0])
                        grid_z[ii, :] = z_i
                    else:
                        grid_z[ii, :] = zd
            else:
                grid_z = griddata((x, y), vrb, (grid_x, grid_y), method=method)
            grid_z[mask_y] = np.nan
            grid_z[mask_x] = np.nan
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
    
    def add_GEO_LST(self):
        lons = self['SC_GEO_LON'].flatten()
        uts = self['SC_DATETIME'].flatten()
        lsts = [ut + datetime.timedelta(seconds=int(lon / 15. * 3600)) for ut, lon in zip(uts, lons)]
        lsts = [lst.hour + lst.minute / 60. + lst.second / 3600. for lst in lsts]
        var = self.add_variable(var_name='SC_GEO_LST')
        var.value = np.array(lsts)[:, np.newaxis]
        var.label = 'LST'
        var.unit = 'h'
        var.depends = self['SC_GEO_LON'].depends
        return var
    
    @staticmethod
    def format_pseudo_lat_label(ax, sector_name, is_integer=True, y_tick_res=15.):
        if is_integer:
            lat_format = '{:4.0f}'
        else:
            lat_format = '{:5.1f}'
    
        y_lim = ax.get_ylim()
        y_max = np.max(y_lim)
        y_min = np.min(y_lim)
        
        yticks_ref = np.arange(-360, 720 + y_tick_res, y_tick_res)
        iii = np.where((yticks_ref>=y_min) & (yticks_ref<=y_max))[0]
        # yticks = ax.get_yticks()
        yticks = yticks_ref[iii]
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
                if lat == -90.:
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
    def smooth_savgol(dts, data, box_pts, t_res, poly_order=1):
        delta_sectime = 1
        sectime, _ = dttool.convert_datetime_to_sectime(dts)
        ind_nan = np.isnan(data)
        sectime_i = np.arange(0, sectime[-1], delta_sectime)
        data_i = np.interp(sectime_i, sectime[~ind_nan], data[~ind_nan])

        window_size = int(np.floor(t_res / delta_sectime * box_pts / 2) * 2 + 1)
        y = sig.savgol_filter(data_i, window_size, poly_order)
        data_new = np.interp(sectime, sectime_i, y)
        data_new[ind_nan] = np.nan
        return data_new
    
    def trajectory_local_unit_vector(self):
        lat_0 = self['SC_GEO_LAT'].flatten()
        lon_0 = self['SC_GEO_LON'].flatten()
        height_0 = self['SC_GEO_ALT'].flatten()
        
        cs_0 = GEOCSpherical(coords={'lat': lat_0, 'lon': lon_0, 'height': height_0})
        phi_0 = cs_0['phi']
        theta_0 = cs_0['theta']
        cs_0 = cs_0.to_cartesian()
        x_0 = cs_0['x']
        y_0 = cs_0['y']
        z_0 = cs_0['z']
        
        dx = np.concatenate(
            (
                [cs_0['x'][1] - cs_0['x'][0]], 
                cs_0['x'][2:] - cs_0['x'][0:-2],
                [cs_0['x'][-1] - cs_0['x'][-2]]
            ),
            axis=0
        )
        dy = np.concatenate(
            (
                [cs_0['y'][1] - cs_0['y'][0]], 
                cs_0['y'][2:] - cs_0['y'][0:-2],
                [cs_0['y'][-1] - cs_0['y'][-2]]
            ),
            axis=0
        )
        dz = np.concatenate(
            (
                [cs_0['z'][1] - cs_0['z'][0]], 
                cs_0['z'][2:] - cs_0['z'][0:-2],
                [cs_0['z'][-1] - cs_0['z'][-2]]
            ),
            axis=0
        )
        
        dv = np.array([dx.flatten(), dy.flatten(), dz.flatten()]).T
        
        v_new = np.empty_like(dv)
        
        for ind, (phi_c, theta_c) in enumerate(zip(phi_0, theta_0)):
            R_1 = np.array([
                [-np.sin(phi_c), -np.cos(phi_c), 0],
                [np.cos(phi_c), -np.sin(phi_c), 0],
                [0, 0, 1]
            ])

            R_2 = np.array([
                [1, 0, 0],
                [0, np.cos(theta_c), -np.sin(theta_c)],
                [0, np.sin(theta_c), np.cos(theta_c)]
            ])
            v_new[ind, :] = dv[ind, :].reshape((1, 3)) @ R_1 @ R_2
            
        
        norm = np.sqrt(v_new[:, 0]**2 + v_new[:, 1]**2 + v_new[:, 2]**2)
        
        v_unit = np.empty_like(v_new)
        v_unit[:, 0] = v_new[:, 1] / norm
        v_unit[:, 1] = v_new[:, 0] / norm
        v_unit[:, 2] = - v_new[:, 2] / norm
        
        return v_unit

         