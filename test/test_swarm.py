import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from loaders.load_uta_gitm_201602_newrun import *
import utilities.datetime_utilities as du

import visualization.time_series as ts
from geospacelab.visualization.mpl.geomap.geopanels import PolarMapPanel as PolarMap
import geospacelab.visualization.mpl.colormaps as cm


def get_swarm_data(dt_fr, dt_to, satID="C"):
    dt_range = [dt_fr, dt_to]

    instr_info1 = {'name': 'SWARM-' + satID, 'assign_key': 'SWARM_ACC'}

    database = 'uta'

    paralist = [
        {'database': database, 'instrument': instr_info1, 'paraname': 'N_n_SC'},
        {'database': database, 'instrument': instr_info1, 'paraname': 'N_n_s450'},
        {'database': database, 'instrument': instr_info1, 'paraname': 'N_n_s500'},
    ]

    paras_layout = [[1, 2]]

    tsObj = ts.TimeSeries(dt_range=dt_range, paralist=paralist, paras_layout=paras_layout, timeline='multiple')

    sc_lat = tsObj.dataObjs['uta_swarm_c_acc'].paras['SC_GEO_LAT']
    sc_lon = tsObj.dataObjs['uta_swarm_c_acc'].paras['SC_GEO_LON']
    sc_lst = tsObj.dataObjs['uta_swarm_c_acc'].paras['SC_GEO_ST']
    sc_datetime = tsObj.dataObjs['uta_swarm_c_acc'].paras['SC_DATETIME']
    rho_n_sc = tsObj.dataObjs['uta_swarm_c_acc'].paras['N_n_SC']

    swarm_data = {
        'sc_lat': sc_lat,
        'sc_lon': sc_lon,
        'sc_lst': sc_lst,
        'sc_datetime': sc_datetime,
        'rho_n_sc': rho_n_sc
    }

    return swarm_data


def show_rho_n(dt_fr, dt_to):
    swarm_data = get_swarm_data(dt_fr, dt_to)
    sc_lon = swarm_data['sc_lon'].flatten()
    sc_lat = swarm_data['sc_lat'].flatten()
    sc_dt = swarm_data['sc_datetime'].flatten()
    rho_n_sc = swarm_data['rho_n_sc'].flatten()

    plt.figure(figsize=(8,8))
    # cs = 'GEO'
    # panel = PolarView(cs='GEO', sytle='lst-fixed', pole='N', lon_c=None, lst_c=0, mlt_c=None, ut=dt_fr, boundary_lat=30., proj_style='Stereographic')
    cs = 'AACGM'
    panel = PolarMap(cs=cs, style='mlt-fixed', pole='N', lon_c=None, lst_c=None, mlt_c=0, ut=dt_fr, boundary_lat=30., proj_style='Stereographic')

    panel.add_subplot(major=True)
    panel.set_extent(boundary_style='circle')

    data = panel.projection.transform_points(ccrs.PlateCarree(), sc_lon, sc_lat)

    x = data[:, 0]
    y = data[:, 1]


    from scipy.stats import linregress
    coef = linregress(x, y)
    a = coef.slope
    b = coef.intercept

    # x1 = np.linspace(np.nanmin(x), np.nanmax(x), num=500)
    # y1 = np.linspace(np.nanmin(y), np.nanmax(y), num=500)
    # x2d, y2d = np.meshgrid(x, y)
    # z2d = griddata(data[:, 0:2], rho_n_sc.flatten(), (x2d, y2d), method='nearest')
    # z2d_dist = np.abs(a*x2d - y2d + b) / np.sqrt(a**2 + 1)
    # z2d = np.where(z2d_dist>1000, np.nan, z2d)
    #
    # im = panel.major_ax.pcolormesh(x2d, y2d, z2d, vmin=2e-14, vmax=20e-13, cmap='gist_ncar')

    #  panel.major_ax.plot(sc_lon, sc_lat, transform=ccrs.Geodetic(), linewidth=0.5, color='k')
    # plt.colorbar(im)

    # xx = np.tile(x, [3, 1])
    #
    # yy = np.concatenate((y[np.newaxis, :]-150000, y[np.newaxis, :], y[np.newaxis, :]+150000), axis=0)
    # zz = np.concatenate((rho_n_sc.T, rho_n_sc.T, rho_n_sc.T), axis=0)
    #
    # zz = griddata(data[:, 0:2], rho_n_sc.flatten(), (xx, yy), method='nearest')
    # im = panel.major_ax.pcolormesh(xx, yy, zz, vmin=2e-14, vmax=20e-13, cmap='gist_ncar')
    #
    # #  panel.major_ax.plot(sc_lon, sc_lat, transform=ccrs.Geodetic(), linewidth=0.5, color='k')
    # plt.colorbar(im)

    #panel.major_ax.plot(data[:,0], data[:,1], linewidth=3)

    from matplotlib.collections import LineCollection
    coords = {'lat': sc_lat, 'lon': sc_lon, 'height': 250.}
    cs_new = panel.cs_transform(cs_fr='GEO', cs_to=cs, coords=coords)
    data = panel.projection.transform_points(ccrs.PlateCarree(), cs_new['lon'], cs_new['lat'])
    x = data[:, 0]
    y = data[:, 1]
    z = rho_n_sc.flatten()
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = plt.Normalize(2e-14, 18e-13)
    lc = LineCollection(segments, cmap='gist_ncar', norm=norm)
    lc.set_array(z)
    lc.set_linewidth(6)
    line = panel.major_ax.add_collection(lc)
    cbar = plt.gcf().colorbar(line, ax=panel.major_ax, pad=0.1, fraction=0.03)
    cbar.set_label('Neutral mass density\n' + r'(kg/m$^{3}$)', rotation=270, labelpad=25)
    #cbaxes = plt.gcf().add_axes([0.8, 0.1, 0.03, 0.8]) 
    #cb = plt.colorbar(panel.major_ax, cax = cbaxes)  

    sectime, dt0 = du.convert_datetime_to_sectime(sc_dt, datetime.datetime(dt_fr.year, dt_fr.month, dt_fr.day))
    sectime_res = 10 * 60
    time_ticks = np.arange(np.floor(sectime[0]/sectime_res)*sectime_res, np.ceil(sectime[-1]/sectime_res)*sectime_res, sectime_res)
    from scipy.interpolate import interp1d
    f = interp1d(sectime, x, fill_value='extrapolate')
    x_time_ticks = f(time_ticks)
    f = interp1d(sectime, y, fill_value='extrapolate')
    y_time_ticks = f(time_ticks)
    panel.major_ax.plot(x_time_ticks, y_time_ticks, '.', markersize=4, color='k')
    for ind, time_tick in enumerate(time_ticks):
        time = dt0 + datetime.timedelta(seconds=time_tick)
        x_time_tick = x_time_ticks[ind]
        y_time_tick = y_time_ticks[ind]

        if x_time_tick < panel._extent[0] or x_time_tick > panel._extent[1]:
            continue
        if y_time_tick < panel._extent[2] or y_time_tick > panel._extent[3]:
            continue

        panel.major_ax.text( x_time_tick, y_time_tick, time.strftime("%d %H:%M"), fontsize=6)

    panel.add_coastlines()
    panel.add_grids()

    plt.gcf().suptitle('Swarm-C neutral mass density\n 2016-02-03T07:00 - 2016-02-03T07:50')
    plt.savefig('test_pho_n_aacgm.png', dpi=300)
    # plt.show()

def show_n_e(dt_fr, dt_to):
    import cdflib
    filepath = "~/tmp/SW_OPER_EFIC_LP_1B_20160203T000000_20160203T235959_0501_MDR_EFI_LP.cdf"
    cf = cdflib.CDF(filepath)
    cf_info = cf.cdf_info()
    n_e = cf.varget(variable='Ne')
    T_e = cf.varget(variable='Te')
    sc_lat = cf.varget(variable='Latitude')
    sc_lon = cf.varget(variable='Longitude')
    timestamp = cf.varget(variable='Timestamp')
    
    dtstrs = cdflib.cdfepoch.encode(timestamp)
    dts = np.empty_like(timestamp, dtype=datetime.datetime)
    for ind, dtstr in enumerate(dtstrs):
        dts[ind] = datetime.datetime.strptime(dtstr+'000', '%Y-%m-%dT%H:%M:%S.%f')
    ind_dt = np.where((dts >= dt_fr) & (dts <= dt_to))[0]
    # times = cdflib.cdfepoch.unixtime(timestamp, to_np=True)
    
    
    sc_lon = sc_lon[ind_dt]
    sc_lat = sc_lat[ind_dt]
    sc_dt = dts[ind_dt]
    rho_n_sc = n_e[ind_dt]

    plt.figure(figsize=(8,8))
    cs = 'GEO'
    panel = PolarMap(
        cs='GEO', style='lst-fixed', pole='N', lon_c=None, lst_c=0, mlt_c=None, ut=dt_fr,
        boundary_lat=0., proj_type='Stereographic')
    # cs = 'AACGM'
    #panel = PolarMap(cs=cs, style='mlt-fixed', pole='N', lon_c=None, lst_c=None, mlt_c=0, ut=dt_fr, boundary_lat=30.,
    #                 proj_type='Stereographic')
    panel.set_extent(boundary_style='circle')

    data = panel.projection.transform_points(ccrs.PlateCarree(), sc_lon, sc_lat)

    x = data[:, 0]
    y = data[:, 1]


    from scipy.stats import linregress
    coef = linregress(x, y)
    a = coef.slope
    b = coef.intercept

    # x1 = np.linspace(np.nanmin(x), np.nanmax(x), num=500)
    # y1 = np.linspace(np.nanmin(y), np.nanmax(y), num=500)
    # x2d, y2d = np.meshgrid(x, y)
    # z2d = griddata(data[:, 0:2], rho_n_sc.flatten(), (x2d, y2d), method='nearest')
    # z2d_dist = np.abs(a*x2d - y2d + b) / np.sqrt(a**2 + 1)
    # z2d = np.where(z2d_dist>1000, np.nan, z2d)
    #
    # im = panel.major_ax.pcolormesh(x2d, y2d, z2d, vmin=2e-14, vmax=20e-13, cmap='gist_ncar')

    #  panel.major_ax.plot(sc_lon, sc_lat, transform=ccrs.Geodetic(), linewidth=0.5, color='k')
    # plt.colorbar(im)

    # xx = np.tile(x, [3, 1])
    #
    # yy = np.concatenate((y[np.newaxis, :]-150000, y[np.newaxis, :], y[np.newaxis, :]+150000), axis=0)
    # zz = np.concatenate((rho_n_sc.T, rho_n_sc.T, rho_n_sc.T), axis=0)
    #
    # zz = griddata(data[:, 0:2], rho_n_sc.flatten(), (xx, yy), method='nearest')
    # im = panel.major_ax.pcolormesh(xx, yy, zz, vmin=2e-14, vmax=20e-13, cmap='gist_ncar')
    #
    # #  panel.major_ax.plot(sc_lon, sc_lat, transform=ccrs.Geodetic(), linewidth=0.5, color='k')
    # plt.colorbar(im)

    #panel.major_ax.plot(data[:,0], data[:,1], linewidth=3)

    from matplotlib.collections import LineCollection
    coords = {'lat': sc_lat, 'lon': sc_lon, 'height': 250.}
    cs_new = panel.cs_transform(cs_fr='GEO', cs_to=cs, coords=coords)
    data = panel.projection.transform_points(ccrs.PlateCarree(), cs_new['lon'], cs_new['lat'])
    x = data[:, 0]
    y = data[:, 1]
    z = rho_n_sc.flatten()
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = colors.LogNorm(vmin=8e3, vmax=1e6)
    lc = LineCollection(segments, cmap=cm.cmap_gist_ncar_modified(), norm=norm)
    lc.set_array(z)
    lc.set_linewidth(6)
    line = panel.major_ax.add_collection(lc)
    cbar = plt.gcf().colorbar(line, ax=panel.major_ax, pad=0.1, fraction=0.03)
    cbar.set_label(r'$n_e$' + '\n' + r'(cm$^{-3}$)', rotation=270, labelpad=25)
    #cbaxes = plt.gcf().add_axes([0.8, 0.1, 0.03, 0.8]) 
    #cb = plt.colorbar(panel.major_ax, cax = cbaxes)  


    sectime, dt0 = du.convert_datetime_to_sectime(sc_dt, datetime.datetime(dt_fr.year, dt_fr.month, dt_fr.day))
    sectime_res = 10 * 60
    time_ticks = np.arange(np.floor(sectime[0]/sectime_res)*sectime_res, np.ceil(sectime[-1]/sectime_res)*sectime_res, sectime_res)
    from scipy.interpolate import interp1d
    f = interp1d(sectime, x, fill_value='extrapolate')
    x_time_ticks = f(time_ticks)
    f = interp1d(sectime, y, fill_value='extrapolate')
    y_time_ticks = f(time_ticks)
    panel.major_ax.plot(x_time_ticks, y_time_ticks, '.', markersize=4, color='k')
    for ind, time_tick in enumerate(time_ticks):
        time = dt0 + datetime.timedelta(seconds=time_tick)
        x_time_tick = x_time_ticks[ind]
        y_time_tick = y_time_ticks[ind]

        if x_time_tick < panel._extent[0] or x_time_tick > panel._extent[1]:
            continue
        if y_time_tick < panel._extent[2] or y_time_tick > panel._extent[3]:
            continue

        panel.major_ax.text( x_time_tick, y_time_tick, time.strftime("%d %H:%M"), fontsize=6)

    panel.add_coastlines()
    panel.add_gridlines()

    plt.gcf().suptitle('Swarm-C electron density\n' + dt_fr.strftime("%Y%m%dT%H%M") + ' - ' + dt_to.strftime("%Y%m%dT%H%M"))
    plt.savefig('swarm_ne_' + cs + '_' + dt_fr.strftime("%Y%m%d_%H%M") + '-' + dt_to.strftime('%H%M'), dpi=300)
    plt.show()
    
def show_T_e(dt_fr, dt_to):
    import cdflib
    filepath = "~/tmp/SW_OPER_EFIC_LP_1B_20160203T000000_20160203T235959_0501_MDR_EFI_LP.cdf"
    cf = cdflib.CDF(filepath)
    cf_info = cf.cdf_info()
    n_e = cf.varget(variable='Ne')
    T_e = cf.varget(variable='Te')
    sc_lat = cf.varget(variable='Latitude')
    sc_lon = cf.varget(variable='Longitude')
    timestamp = cf.varget(variable='Timestamp')
    
    dtstrs = cdflib.cdfepoch.encode(timestamp)
    dts = np.empty_like(timestamp, dtype=datetime.datetime)
    for ind, dtstr in enumerate(dtstrs):
        dts[ind] = datetime.datetime.strptime(dtstr+'000', '%Y-%m-%dT%H:%M:%S.%f')
    ind_dt = np.where((dts >= dt_fr) & (dts <= dt_to))[0]
    # times = cdflib.cdfepoch.unixtime(timestamp, to_np=True)
    
    
    sc_lon = sc_lon[ind_dt]
    sc_lat = sc_lat[ind_dt]
    sc_dt = dts[ind_dt]
    rho_n_sc = T_e[ind_dt]

    plt.figure(figsize=(8,8))
    panel = PolarView(cs='GEO', pole='N', lon_c=None, lst_c=0, ut=dt_fr, boundary_lat=0., proj_style='Stereographic')
    panel.add_subplot(major=True)
    panel.set_extent(boundary_style='circle')

    data = panel.projection.transform_points(ccrs.PlateCarree(), sc_lon, sc_lat)

    x = data[:, 0]
    y = data[:, 1]


    from scipy.stats import linregress
    coef = linregress(x, y)
    a = coef.slope
    b = coef.intercept

    # x1 = np.linspace(np.nanmin(x), np.nanmax(x), num=500)
    # y1 = np.linspace(np.nanmin(y), np.nanmax(y), num=500)
    # x2d, y2d = np.meshgrid(x, y)
    # z2d = griddata(data[:, 0:2], rho_n_sc.flatten(), (x2d, y2d), method='nearest')
    # z2d_dist = np.abs(a*x2d - y2d + b) / np.sqrt(a**2 + 1)
    # z2d = np.where(z2d_dist>1000, np.nan, z2d)
    #
    # im = panel.major_ax.pcolormesh(x2d, y2d, z2d, vmin=2e-14, vmax=20e-13, cmap='gist_ncar')

    #  panel.major_ax.plot(sc_lon, sc_lat, transform=ccrs.Geodetic(), linewidth=0.5, color='k')
    # plt.colorbar(im)

    # xx = np.tile(x, [3, 1])
    #
    # yy = np.concatenate((y[np.newaxis, :]-150000, y[np.newaxis, :], y[np.newaxis, :]+150000), axis=0)
    # zz = np.concatenate((rho_n_sc.T, rho_n_sc.T, rho_n_sc.T), axis=0)
    #
    # zz = griddata(data[:, 0:2], rho_n_sc.flatten(), (xx, yy), method='nearest')
    # im = panel.major_ax.pcolormesh(xx, yy, zz, vmin=2e-14, vmax=20e-13, cmap='gist_ncar')
    #
    # #  panel.major_ax.plot(sc_lon, sc_lat, transform=ccrs.Geodetic(), linewidth=0.5, color='k')
    # plt.colorbar(im)

    #panel.major_ax.plot(data[:,0], data[:,1], linewidth=3)

    from matplotlib.collections import LineCollection

    data = panel.projection.transform_points(ccrs.PlateCarree(), sc_lon, sc_lat)
    x = data[:, 0]
    y = data[:, 1]
    z = rho_n_sc.flatten()
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = plt.Normalize(500, 4000)
    # norm = colors.LogNorm(vmin=5e2, vmax=1e6)
    lc = LineCollection(segments, cmap='gist_ncar', norm=norm)
    lc.set_array(z)
    lc.set_linewidth(6)
    line = panel.major_ax.add_collection(lc)
    cbar = plt.gcf().colorbar(line, ax=panel.major_ax, pad=0.1, fraction=0.03)
    cbar.set_label(r'$T_e$' + '\n' + r'(K)', rotation=270, labelpad=25)
    #cbaxes = plt.gcf().add_axes([0.8, 0.1, 0.03, 0.8]) 
    #cb = plt.colorbar(panel.major_ax, cax = cbaxes)  


    sectime, dt0 = du.convert_datetime_to_sectime(sc_dt, datetime.datetime(dt_fr.year, dt_fr.month, dt_fr.day))
    sectime_res = 10 * 60
    time_ticks = np.arange(np.floor(sectime[0]/sectime_res)*sectime_res, np.ceil(sectime[-1]/sectime_res)*sectime_res, sectime_res)
    from scipy.interpolate import interp1d
    f = interp1d(sectime, x, fill_value='extrapolate')
    x_time_ticks = f(time_ticks)
    f = interp1d(sectime, y, fill_value='extrapolate')
    y_time_ticks = f(time_ticks)
    panel.major_ax.plot(x_time_ticks, y_time_ticks, '.', markersize=4, color='k')
    for ind, time_tick in enumerate(time_ticks):
        time = dt0 + datetime.timedelta(seconds=time_tick)
        x_time_tick = x_time_ticks[ind]
        y_time_tick = y_time_ticks[ind]

        if x_time_tick < panel._extent[0] or x_time_tick > panel._extent[1]:
            continue
        if y_time_tick < panel._extent[2] or y_time_tick > panel._extent[3]:
            continue

        panel.major_ax.text( x_time_tick, y_time_tick, time.strftime("%d %H:%M"), fontsize=6)

    panel.add_coastlines()
    panel.add_grids()

    plt.gcf().suptitle('Swarm-C electron temperature\n 2016-02-03T07:00 - 2016-02-03T07:50')
    plt.savefig('test_T_e.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    dt_fr = datetime.datetime(2016, 2, 3, 0, 40)
    dt_to = datetime.datetime(2016, 2, 3, 1, 40)

    # show_rho_n(dt_fr, dt_to)
    
    show_n_e(dt_fr, dt_to)
    
    # show_T_e(dt_fr, dt_to)