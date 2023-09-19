import datetime
import numpy as np


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

