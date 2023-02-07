# Licensed under the BSD 3-Clause License
# Copyright (C) 2021 GeospaceLab (geospacelab)
# Author: Lei Cai, Space Physics and Astronomy, University of Oulu

__author__ = "Lei Cai"
__copyright__ = "Copyright 2021, GeospaceLab"
__license__ = "BSD-3-Clause License"
__email__ = "lei.cai@oulu.fi"
__docformat__ = "reStructureText"


import numpy
import bisect

from datetime import timedelta
from datetime import datetime
from datetime import date
from dateutil.relativedelta import relativedelta
import cftime


def convert_date_to_datetime(mydate):
    mydatetime = datetime.combine(mydate, datetime.min.time())
    return mydatetime


def convert_datetime_to_date(mydatetime):
    mydate = mydatetime.date()
    return mydate


def get_start_of_the_day(dt):
    dt0 = datetime(dt.year, dt.month, dt.day)
    return dt0


def get_end_of_the_day(dt):
    dt0 = datetime(dt.year, dt.month, dt.day, 23, 59, 59)
    return dt0


def convert_datevec_to_datetime(datevec):
    order = ['year', 'month', 'day', 'hour', 'minute', 'second']
    dt_str = "{:04d}".format(datevec[0]) + "{:02d}".format(datevec[1]) + \
             "{:02d}".format(datevec[2]) + "{:02d}".format(datevec[3]) + \
             "{:02d}".format(datevec[4]) + "{:02d}".format(datevec[5])
    dt0 = datetime.strptime(dt_str, "%Y%m%d%H%M%S")
    return dt0


def convert_datetime_to_datevec(dt0):
    dt_vec = [dt0.year, dt0.month, dt0.day, dt0.hour, dt0.minute, dt0.second]
    return dt_vec


def convert_datetime_to_sectime(dts, dt0=None):
    dts_type = type(dts)
    if dts_type is datetime:
        dts = [dts]

    if isinstance(dts, (list, tuple)):
        dts = numpy.array(dts)
    shape = dts.shape

    if dt0 is None:
        dt0 = get_start_of_the_day(dts.flatten()[0])

    dt_delta = dts.flatten() - dt0
    sectime = [dt_temp.total_seconds() for dt_temp in dt_delta]

    if dts_type is list:
        return sectime, dt0
    elif dts_type is tuple:
        return tuple(sectime), dt0
    else:
        sectime = numpy.array(sectime).reshape(shape)
        return sectime, dt0


def get_diff_months(dt1, dt2):
    diff_months = ((dt2.year - dt1.year) * 12) + dt2.month - dt1.month
    return diff_months


def get_diff_days(dt1, dt2):
    dtdelta = datetime(dt2.year, dt2.month, dt2.day) - datetime(dt1.year, dt1.month, dt1.day)
    return int(numpy.ceil(dtdelta.total_seconds() / 86400))


def get_first_day_of_month(dt):
    if type(dt) is date:
        mydate = dt
        datetype = 0
    elif issubclass(dt.__class__, datetime):
        mydate = convert_datetime_to_date(dt)
        datetype = 1
    this_month = mydate - timedelta(days=mydate.day - 1)
    if datetype == 0:
        return this_month
    if datetype == 1:
        return convert_date_to_datetime(this_month)


def get_last_day_of_month(dt, end=False):
    if type(dt) is date:
        mydate = dt
        datetype = 0
    elif issubclass(dt.__class__, datetime):
        mydate = convert_datetime_to_date(dt)
        datetype = 1
    next_month = mydate.replace(day=28) + timedelta(days=4)  # this will never fail
    end_month = next_month - timedelta(days=next_month.day)

    if datetype == 0:
        return end_month
    if datetype == 1:
        dt = convert_date_to_datetime(end_month)
        if end:
            dt = dt + timedelta(seconds=86400-1)
        return dt


def get_next_of_month(dt):
    if type(dt) is date:
        mydate = dt
        datetype = 0
    elif issubclass(dt.__class__, datetime):
        mydate = convert_datetime_to_date(dt)
        datetype = 1
    next_month = mydate.replace(day=28) + timedelta(days=4)  # this will never fail
    next_month = next_month - timedelta(days=next_month.day - 1)
    if datetype == 0:
        return next_month
    if datetype == 1:
        return convert_date_to_datetime(next_month)


def get_next_n_months(dt, nm):
    if type(dt) is date:
        mydate = dt
        datetype = 0
    elif issubclass(dt.__class__, datetime):
        mydate = convert_datetime_to_date(dt)
        datetype = 1
    this_month = get_first_day_of_month(mydate)
    next_n_month = this_month + relativedelta(months=nm)
    if datetype == 0:
        return next_n_month
    if datetype == 1:
        return convert_date_to_datetime(next_n_month)


def convert_datetime_to_matlabdn(dts):
    type_in = type(dts)
    dts = numpy.array(dts)
    dns = numpy.empty_like(dts)
    for ind, dt in enumerate(dts.flatten()):

        mdn = dt + timedelta(days=366)
        frac_seconds = (dt - datetime(dt.year, dt.month, dt.day, 0, 0, 0)).seconds / (24.0 * 60.0 * 60.0)
        frac_microseconds = dt.microsecond / (24.0 * 60.0 * 60.0 * 1000000.0)
        dns[ind] = mdn.toordinal() + frac_seconds + frac_microseconds
    if type_in is datetime:
        return dns[0]
    elif type_in is list:
        return dns.tolist()
    elif 'numpy' in str(type_in):
        return dns.astype(numpy.double)

def convert_matlabdn_to_datetime(dns):
    type_in = type(dns)
    dns = numpy.array(dns)
    dts = numpy.empty_like(dns, dtype=datetime)
    for ind, dn in enumerate(dns.flatten()):
        dts[ind] = datetime.fromordinal(int(dn)) + timedelta(days=dn % 1) - timedelta(days=366)

    if type_in is datetime:
        return dts[0]
    elif type_in is list:
        return dts.tolist()
    elif 'numpy' in str(type_in):
        return dts


def get_doy(dts, year=None, decimal=False):
    type_in = type(dts)
    if type_in is datetime:
        dts = [dts]
    dts = numpy.array(dts)
    doys = numpy.empty_like(dts)

    if year is not None:
        dt0 = datetime(year, 1, 1, 0, 0, 0)
    else:
        dt0 = datetime(dts[0].year, 1, 1, 0, 0, 0)

    for ind, dt in enumerate(dts.flatten()):
        delta_dt = dt - dt0
        doys[ind] = delta_dt.total_seconds()/86400. + 1
    if not decimal:
        doys = numpy.floor(doys)
    if type_in is datetime:
        return doys[0]
    elif type_in is list:
        return doys.tolist()
    elif 'numpy' in str(type_in):
        return doys


def convert_doy_to_datetime(year, doys):
    type_in = type(doys)
    if isinstance(doys, (int, float)):
        doys = [doys]
    doys = numpy.array(doys)
    dts = numpy.empty_like(doys, dtype=datetime)
    dt0 = datetime(year, 1, 1, 0, 0, 0)
    total_seconds = (doys - 1) * 86400.
    for ind, total_second in enumerate(total_seconds.flatten()):
        delta_dt = timedelta(seconds=total_second)
        dts[ind] = dt0 + delta_dt

    if type_in in (int, float):
        return dts[0]
    elif type_in is list:
        return dts.tolist()
    elif 'numpy' in str(type_in):
        return dts


def convert_unix_time_to_datetime(times):
    type_in = type(times)

    ts = numpy.array(times)
    dts = numpy.empty_like(ts, dtype=datetime)

    for ind, t in enumerate(ts.flatten()):
        dts[ind] = datetime.utcfromtimestamp(t)

    if type_in in (int, float):
        return dts[0]
    elif type_in is list:
        return dts.tolist()
    elif 'numpy' in str(type_in):
        return dts


def convert_unix_time_to_datetime_cftime(times):
    type_in = type(times)

    ts = numpy.array(times)
    dts = numpy.empty_like(ts, dtype=datetime)

    dts = numpy.array(
        cftime.num2date(ts.flatten(),
                        units='seconds since 1970-01-01 00:00:00.0',
                        only_use_cftime_datetimes=False,
                        only_use_python_datetimes=True)
    )

    if type_in in (int, float):
        return dts[0]
    elif type_in is list:
        return dts.tolist()
    elif 'numpy' in str(type_in):
        return dts.reshape(times.shape)


def convert_gps_time_to_datetime(times, weeks=None):
    
    type_in = type(times)

    ts = numpy.array(times).flatten()
    if weeks is None:
        weeks = numpy.zeros_like(ts)
    else:
        weeks = numpy.array(weeks).flatten()

    gps_seconds = weeks * _SECONDS_PER_WEEK + times
    gps_seconds_add = numpy.array([bisect.bisect_left(_LEAP_SECONDS_GPS_TIME, gs) for gs in gps_seconds])
    dts = numpy.array([_GPS_DATETIME_0 + timedelta(seconds=sec) for sec in (gps_seconds - gps_seconds_add)])

    if type_in in (int, float):
        return dts[0]
    elif type_in is list:
        return dts.tolist()
    elif 'numpy' in str(type_in):
        return dts.reshape(times.shape)

def convert_datetime_to_gps_times(times: datetime, with_weeks=False):
    
    type_in = type(times)

    ts = numpy.array(times).flatten()
    
    ts_seconds = numpy.array([ (t - _GPS_DATETIME_0).total_seconds() for t in ts])
    print(ts_seconds)
    
    leap_seconds_dates = [datetime(i[0], i[1], i[2], 23, 59, 59) for i in _LEAP_SECONDS_DATES_GPS] 

    gps_seconds = numpy.array([sec + bisect.bisect_left(leap_seconds_dates, t) for sec, t in zip(ts_seconds, ts)])
    print(gps_seconds)
    if with_weeks:
        weeks = numpy.floor(gps_seconds / _SECONDS_PER_WEEK)
        gps_seconds = gps_seconds % _SECONDS_PER_WEEK
        if type_in is datetime:
            return gps_seconds[0], weeks[0]
        elif type_in is list: 
            return gps_seconds.tolist(), weeks.tolist()
        elif 'numpy' in str(type_in):
            return gps_seconds.reshape(times.shape), weeks.reshape(times.shape)
    else:
        if type_in is datetime:
            return gps_seconds[0]
        elif type_in is list: 
            return gps_seconds.tolist()
        elif 'numpy' in str(type_in):
            return gps_seconds.reshape(times.shape)
        

_GPS_DATETIME_0 = datetime(1980, 1, 6)
_SECONDS_PER_WEEK = 604800.0
_LEAP_SECONDS_DATES = [
    (1972, 6, 30), (1972, 12, 31), (1973, 12, 31), (1974, 12, 31), (1975, 12, 31), 
    (1976, 12, 31), (1977, 12, 31), (1978, 12, 31), (1979, 12, 31), (1980, 6, 30), 
    (1982, 6, 30), (1983, 6, 30), (1985, 6, 30), (1987, 12, 31), (1989, 12, 31), 
    (1990, 12, 31), (1992, 6, 30), (1993, 6, 30),  (1994, 6, 30), (1995, 12, 31),
    (1997, 6, 30), (1998, 12, 31), (2005, 12, 31), (2008, 12, 31), (2012, 6, 30),
    (2015, 6, 30), (2016, 12, 31), 
]

_LEAP_SECONDS_DATES_GPS = [
    (1980, 6, 30), (1982, 6, 30), (1983, 6, 30), (1985, 6, 30), (1987, 12, 31), 
    (1989, 12, 31), (1990, 12, 31), (1992, 6, 30), (1993, 6, 30),  (1994, 6, 30), 
    (1995, 12, 31), (1997, 6, 30), (1998, 12, 31), (2005, 12, 31), (2008, 12, 31), 
    (2012, 6, 30), (2015, 6, 30), (2016, 12, 31), 
]

_LEAP_SECONDS_COUNT_GPS = [
    0., 1., 2., 3., 4., 
    5., 6., 7., 8., 9., 10., 
    11., 12., 13., 14., 15.,
    16., 17., 18., 
]

# gps_seconds = [(datetime(i[0], i[1], i[2], 23, 59, 59) - _GPS_DATETIME_0).total_seconds() + j - 1 for i, j in zip(_LEAP_SECONDS_DATES, _LEAP_SECONDS_COUNT_GPS)]

_LEAP_SECONDS_GPS_TIME = [
    15292799.0, 78364800.0, 109900801.0, 173059202.0, 252028803.0, 
    315187204.0, 346723205.0, 393984006.0, 425520007.0, 457056008.0, 
    504489609.0, 551750410.0, 599184011.0, 820108812.0, 914803213.0, 
    1025136014.0, 1119744015.0, 1167264016.0
]

