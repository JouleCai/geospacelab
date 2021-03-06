import numpy

from datetime import timedelta
from datetime import datetime
from datetime import date
from dateutil.relativedelta import relativedelta


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
    elif type(dt) is datetime:
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
    elif type(dt) is datetime:
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
    elif type(dt) is datetime:
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
    elif type(dt) is datetime:
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
    if isinstance(type_in, (int, float)):
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

