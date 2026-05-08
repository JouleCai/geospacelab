import numpy as np
from scipy.signal import argrelextrema
import datetime
from scipy.interpolate import griddata, interp1d, CubicSpline
from scipy.signal import butter, lfilter, freqz
import scipy.signal as sig

from geospacelab.datahub import DatasetUser
from geospacelab.toolbox.utilities import pydatetime as dttool
import geospacelab.toolbox.utilities.numpymath as npmath
import geospacelab.toolbox.utilities.interpolation as gsl_interp
import geospacelab.toolbox.utilities.binning as gsl_binning
import geospacelab.toolbox.utilities.pylogging as mylog
import geospacelab.cs as gsl_cs

from geospacelab.observatory.orbit.sc_orbit import OrbitPosition_SSCWS


def conjunction_leo_to_site(
    dt_fr = None,
    dt_to = None,
    sat_id = None,
    glat_site = None,
    glon_site = None,
    alt_site = None,
    el_lim = 75.,
    el_lim_0 = 15.,
    print_conj_list = False,
):
    el_lim_0 = 15. # Search from 30 degree, then check if the satellite is within 75 degree. This is to save time.
    
    conj_list_dtype = np.dtype([
        ('DATETIME', 'O'),
        ('DATETIME_0', 'O'),
        ('DATETIME_1', 'O'),
        ('SC_GEO_LAT', 'f8'),
        ('SC_GEO_LAT_0', 'f8'),
        ('SC_GEO_LAT_1', 'f8'),
        ('SC_GEO_LON', 'f8'),
        ('SC_GEO_LON_0', 'f8'),
        ('SC_GEO_LON_1', 'f8'),
        ('SC_GEO_ALT', 'f8'),
        ('SC_GEO_ALT_0', 'f8'),
        ('SC_GEO_ALT_1', 'f8'),
        ('EL', 'f8'),
        ('EL_0', 'f8'),
        ('EL_1', 'f8'),
        ('AZ', 'f8'),
        ('AZ_0', 'f8'),
        ('AZ_1', 'f8'),
        ('DURATION', 'f8'),
        ('Distance', 'f8'),
        ('Distance_0', 'f8'),
        ('Distance_1', 'f8'),
        ('GEO_LAT_LOC', 'f8'),
        ('GEO_LON_LOC', 'f8'),
        ('GEO_ALT_LOC', 'f8'),
    ])
    
    conj_list = np.array([], dtype=conj_list_dtype)
    
    ds_leo = OrbitPosition_SSCWS(dt_fr-datetime.timedelta(hours=3), dt_to+datetime.timedelta(hours=3), sat_id=sat_id)
    glats_leo = ds_leo['SC_GEO_LAT'].flatten()
    glons_leo = ds_leo['SC_GEO_LON'].flatten()
    galts_leo = ds_leo['SC_GEO_ALT'].flatten()
    xs_leo = ds_leo['SC_GEO_X'].flatten()
    ys_leo = ds_leo['SC_GEO_Y'].flatten()
    zs_leo = ds_leo['SC_GEO_Z'].flatten()
    dts_leo = ds_leo['SC_DATETIME'].flatten()
    secttimes_leo, dt0 = dttool.convert_datetime_to_sectime(dts_leo)
    
    els_leo_0 = calc_el(glat_site, glon_site, alt_site, glats_leo, glons_leo, galts_leo)
    inds_el_0 = np.where(els_leo_0>=el_lim_0)[0]
    
    diff_inds_el = np.diff(inds_el_0)
    inds_seg_el = np.where(diff_inds_el>1)[0]
    
    if inds_seg_el.size == 0:
        segment_indices_el = [inds_el_0]
    else:
        segment_boundries_el = inds_seg_el + 1
        segment_indices_el = np.split(inds_el_0, segment_boundries_el)
    
    for inds in segment_indices_el:
        if inds[0] > 1:
            inds = np.insert(inds, 0, inds[0]-1)
        if inds[-1] < len(glats_leo)-2:
            inds = np.append(inds, inds[-1]+1)
        glats_leo_seg = glats_leo[inds]
        glons_leo_seg = glons_leo[inds]
        galts_leo_seg = galts_leo[inds]
        
        xs_leo_seg = xs_leo[inds]
        ys_leo_seg = ys_leo[inds]
        zs_leo_seg = zs_leo[inds]
        
        sectimes_leo_seg = secttimes_leo[inds]
        sectimes_new = np.arange(sectimes_leo_seg[0], sectimes_leo_seg[-1], 1.)
        
        
        f_x = CubicSpline(sectimes_leo_seg, xs_leo_seg)
        x_i = f_x(sectimes_new)
        f_y = CubicSpline(sectimes_leo_seg, ys_leo_seg)
        y_i = f_y(sectimes_new)
        f_z = CubicSpline(sectimes_leo_seg, zs_leo_seg)
        z_i = f_z(sectimes_new)
        
        
        cs = gsl_cs.GEOCCartesian(
            coords={'x': x_i, 'y': y_i, 'z': z_i, 'x_unit': 'km', 'y_unit': 'km', 'z_unit': 'km'}
        )
        cs_new = cs.to_spherical()
        
        glats_leo_seg_new = cs_new['lat']
        glons_leo_seg_new = cs_new['lon']
        galts_leo_seg_new = cs_new['r'] - 6371.2
        
        els_leo_seg_new = calc_el(glat_site, glon_site, alt_site, glats_leo_seg_new, glons_leo_seg_new, galts_leo_seg_new)
        azs_leo_seg_new = calc_az(glat_site, glon_site, alt_site, glats_leo_seg_new, glons_leo_seg_new, galts_leo_seg_new)
        Ds_leo_seg_new = calc_big_circle_distance(glat_site, glon_site, glats_leo_seg_new, glons_leo_seg_new, galts_leo_seg_new)
        
        inds_el_seg = np.where(els_leo_seg_new>=el_lim)[0]
        if inds_el_seg.size > 0:
            sectimes_conj = sectimes_new[inds_el_seg]
            dts_conj = np.array([dt0 + datetime.timedelta(seconds=sect) for sect in sectimes_conj])
            glats_conj = glats_leo_seg_new[inds_el_seg]
            glons_conj = glons_leo_seg_new[inds_el_seg]
            galts_conj = galts_leo_seg_new[inds_el_seg]
            els_conj = els_leo_seg_new[inds_el_seg]
            azs_conj = azs_leo_seg_new[inds_el_seg]
            Ds_conj = Ds_leo_seg_new[inds_el_seg]
            ind_nearest = np.argmax(els_conj)
            
            data_add = (
                dts_conj[ind_nearest], dts_conj[0], dts_conj[-1],
                glats_conj[ind_nearest], glats_conj[0], glats_conj[-1],
                glons_conj[ind_nearest], glons_conj[0], glons_conj[-1],
                galts_conj[ind_nearest], galts_conj[0], galts_conj[-1],
                els_conj[ind_nearest], els_conj[0], els_conj[-1],
                azs_conj[ind_nearest], azs_conj[0], azs_conj[-1],
                sectimes_conj[-1] - sectimes_conj[0],
                Ds_conj[ind_nearest], Ds_conj[0], Ds_conj[-1],
                glat_site, glon_site, alt_site,
            )
            conj_list = np.append(conj_list, np.array(data_add, dtype=conj_list.dtype))
        
    if print_conj_list:
        print_conjunction_list_leo_to_site(conj_list)
            
    return conj_list


def filtering_by_instant_times(conj_list, dts_instant, time_window=None):
    # Filter the conjunction data by the instant times. Only keep the conjunctions that have the highest elevation within 10 minutes of the instant times.
    dts_instant = np.array(dts_instant)
    
    dts_conj = conj_list['DATETIME']
    
    if time_window is not None:
        dts_conj_0 = dts_conj - datetime.timedelta(seconds=time_window/2)
        dts_conj_1 = dts_conj + datetime.timedelta(seconds=time_window/2)
    else:
        dts_conj_0 = conj_list['DATETIME_0']
        dts_conj_1 = conj_list['DATETIME_1']
    
    inds_keep = []
    for i, (dt_0, dt_1) in enumerate(zip(dts_conj_0, dts_conj_1)):
        if np.any((dts_instant >= dt_0) & (dts_instant <= dt_1)):
            inds_keep.append(i)
    
    return conj_list[inds_keep]


def filtering_by_time_ranges(conj_list, dt_ranges, time_window=None):
    # Filter the conjunction data by the time ranges. Only keep the conjunctions that have the highest elevation within the time ranges.
    dt_ranges = np.array(dt_ranges)
    
    dts_conj = conj_list['DATETIME']
    
    if time_window is not None:
        dts_conj_0 = dts_conj - datetime.timedelta(seconds=time_window/2)
        dts_conj_1 = dts_conj + datetime.timedelta(seconds=time_window/2)
    else:   
        dts_conj_0 = conj_list['DATETIME_0']
        dts_conj_1 = conj_list['DATETIME_1']
    
    inds_keep = []
    for i, (dt_0, dt_1) in enumerate(zip(dts_conj_0, dts_conj_1)):
        inds_invalid = np.where((dt_1 < dt_ranges[0]) | (dt_0 > dt_ranges[1]))[0]
        if len(inds_invalid) < len(dt_ranges):
            inds_keep.append(i)    
    return conj_list[inds_keep]


def print_conjunction_list_leo_to_site(conj_list):
    if len(conj_list) == 0:
        print("No conjunction found.")
        return conj_list
    
    mylog.simpleinfo.info(f"{len(conj_list)} conjunction(s) found:")
    mylog.simpleinfo.info(
        f"{'Index':<6} {'Time (Start)':<20} {'EL (Start)':<10} {'AZ (Start)':<10} {'D (Start)':<10} " + \
            f"{'Time (Highest)':<25} {'EL (Highest)':<15} {'AZ (Highest)':<15} {'D (Highest)':<15} " + \
            f"{'Time (End)':<20} {'EL (End)':<10} {'AZ (End)':<10} {'D (End)':<10} {'Duration(s)':<10}"
        )
    for i in range(len(conj_list)):
        mylog.simpleinfo.info(
            f"{i:<6} {conj_list['DATETIME_0'][i].strftime('%Y-%m-%d %H:%M:%S'):<20} {conj_list['EL_0'][i]:<10.2f} {conj_list['AZ_0'][i]:<10.2f} {conj_list['Distance_0'][i]:<10.2f} " + \
            f"{conj_list['DATETIME'][i].strftime('%Y-%m-%d %H:%M:%S'):<25} {conj_list['EL'][i]:<15.2f} {conj_list['AZ'][i]:<15.2f} {conj_list['Distance'][i]:<15.2f} " + \
            f"{conj_list['DATETIME_1'][i].strftime('%Y-%m-%d %H:%M:%S'):<20} {conj_list['EL_1'][i]:<10.2f} {conj_list['AZ_1'][i]:<10.2f} {conj_list['Distance_1'][i]:<10.2f} {conj_list['DURATION'][i]:<10.2f}"
        )


def calc_big_circle_distance(glat_0, glon_0, glat, glon, alt):
    """Calculate Big Circle Distance using Haversin Formular

    Parameters
    ----------
    glat_0 : float
        Latitude of the initial point in degrees
    glon_0 : float
        Longitude of the initial point in degrees
    glat : float
        Latitude of the target point in degrees
    glon : float
        Longitude of the target point in degrees
    alt : float
        Altitude of the target point in km
    """
    
    R_E = 6371.2 # Earth radius in km   
    factor = np.pi / 180.
    glat_0 = glat_0 * factor
    glon_0 = glon_0 * factor
    glat = glat * factor
    glon = glon * factor
    
    delta_glat = glat - glat_0
    delta_glon = glon - glon_0
    
    a = np.sin(delta_glat/2)**2 + np.cos(glat_0)*np.cos(glat)*np.sin(delta_glon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    d = R_E * c
    return d
        
def calc_el(glat_0, glon_0, alt_0, glat, glon, alt):
    # Calculate the elevation angle of the satellite at (glat, glon, alt) as seen from the site at (glat_0, glon_0, alt_0)
    # All in degree and km
    # Reference: https://en.wikipedia.org/wiki/Geographic_coordinate_system#Distance_calculations
    R_E = 6371.2 # Earth radius in km
    
    factor = np.pi / 180.
    glat_0 = glat_0 * factor
    glon_0 = glon_0 * factor
    glat = glat * factor
    glon = glon * factor
    
    theta = np.arccos(
        np.sin(glat)*np.sin(glat_0) + 
        np.cos(glat)*np.cos(glat_0)*np.cos(np.abs((glon-glon_0)))
        )
    
    r = R_E + alt
    r_0 = R_E + alt_0

    s = np.sqrt(r**2 + r_0**2 - 2*r*r_0*np.cos(theta))
    cos_el = (r_0**2 + s**2 - r**2) / (2*r_0*s)

    el = np.arccos(cos_el) / factor - 90
    
    return el
    
def calc_az(glat_0, glon_0, alt_0, glat, glon, alt):
    # Calculate the azimuth angle of the satellite at (glat, glon, alt) as seen from the site at (glat_0, glon_0, alt_0)
    # All in degree and km
    # Reference: https://en.wikipedia.org/wiki/Geographic_coordinate_system#Distance_calculations
    factor = np.pi / 180.
    glat_0 = glat_0 * factor
    glon_0 = glon_0 * factor
    glat = glat * factor
    glon = glon * factor

    y = np.sin((glon-glon_0)*factor) * np.cos(glat*factor)
    x = np.cos(glat_0*factor)*np.sin(glat*factor) - np.sin(glat_0*factor)*np.cos(glat*factor)*np.cos((glon-glon_0)*factor)
    az = np.arctan2(y, x) / factor
    az = (az + 360) % 360
    
    return az


if __name__ == "__main__":
    
    el = calc_el(65, 25, 0, 65, 15, 500)
    az = calc_az(65, 25, 0, 65, 15, 500)
    
    # print(el, az)
    
    conj_data = conjunction_leo_to_site(
        dt_fr = datetime.datetime(2025, 9, 2, 0, 0, 0),
        dt_to = datetime.datetime(2025, 9, 3, 0, 0, 0),
        glat_site = 69.6265,
        glon_site = 19.06,
        alt_site = 0,
        sat_id = 'swarma',
        print_list = True,
    )
     