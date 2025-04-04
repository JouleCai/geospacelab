import numpy as np
import datetime




def get_sub_solar_point(dts):


    ye, mo, da, ho, mi, se = utc
    ta = pi * 2
    ut = ho + mi / 60 + se / 3600
    t = 367 * ye - 7 * (ye + (mo + 9) // 12) // 4
    dn = t + 275 * mo // 9 + da - 730531.5 + ut / 24
    sl = dn * 0.01720279239 + 4.894967873
    sa = dn * 0.01720197034 + 6.240040768
    t = sl + 0.03342305518 * sin(sa)
    ec = t + 0.0003490658504 * sin(2 * sa)
    ob = 0.4090877234 - 0.000000006981317008 * dn
    st = 4.894961213 + 6.300388099 * dn
    ra = atan2(cos(ob) * sin(ec), cos(ec))
    de = asin(sin(ob) * sin(ec))
    la = degrees(de)
    lo = degrees(ra - st) % 360
    lo = lo - 360 if lo > 180 else lo
    return [round(la, 6), round(lo, 6)]




def solar_declination(dt, degrees=False):
    factor = np.pi / 180.
    sd_SS = - 23.4333333 * factor   # Declination angle at southern summer solstice.

    dt_0 = datetime.datetime(dt.year, dt.month, dt.day, 0, )
    N = (dt - dt_0).total_seconda() / 86400.
    sin_EL = np.cos(
        2 * np.pi / 365.24 * (N + 10) + 2 * 0.0167 * np.sin(2 * np.pi/365.24 * (N - 2))
    )

    sd = np.arcsin(
        np.sin(sd_SS) * sin_EL
    )

    if degrees:
        sd = sd / factor
    return sd


def subsolar_point(dt, degrees=False):
    phi = solar_declination(dt, degrees)

    lamda = - 15 * ((dt.hour + dt.minute / 60 + dt.second / 3600) - 12. + equation_of_time(dt))

    
def equation_of_time(dt):
    return

