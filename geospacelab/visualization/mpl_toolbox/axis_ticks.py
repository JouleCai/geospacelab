import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import datetime as dt
from matplotlib.ticker import LogFormatterSciNotation
import geospacelab.toolbox.utilities.pydatetime as dttool
import os


def set_timeline(dt_start, dt_stop, **kwargs):
    mdlocators = {
        1:   'mdates.MicrosecondLocator',
        2:   'mdates.SecondLocator',
        3:   'mdates.MinuteLocator',
        4:   'mdates.HourLocator',
        5:   'mdates.DayLocator',
        6:   'mdates.MonthLocator',
        7:   'mdates.YearLocator',
    }
    default_second = {
        'range':        [1, 2, 3, 5, 10, 15, 20, 30, 40, 60],
        'majorscale':   [10/60, 15/60, 30/60, 1, 2, 3, 4, 5, 5, 10],
        'minorscale':   [2/60, 2/60, 5/60, 10/60, 20/60, 30/60, 1, 1, 1, 2],
        'scale':        1000
    }
    default_minute = {
        'range':        [1, 2, 3, 5, 8, 15, 20, 30, 40, 60],
        'majorscale':   [15/60, 20/60, 30/60, 1, 2, 3, 4, 5, 10],
        'minorscale':   [3/60, 5/60, 5/60, 10/60, 20/60, 30/60, 1, 1, 2],
        'scale': 60
    }
    default_hour = {
        'range':        [1, 2, 3, 5, 8, 12, 18, 24],
        'majorscale':   [15/60, 20/60, 30/60, 1, 2, 3, 4],
        'minorscale':   [3/60, 5/60, 5/60, 10/60, 20/60, 30/60, 1],
        'scale': 60
    }

    default_day = {
        'range':      [1, 2, 3, 5, 8, 14, 21, 30, 50, 80, 122],
        'majorscale': [6/24, 8/24, 12/24, 1, 2, 3, 5, 10, 15],
        'minorscale': [1/24, 2/24, 3/24, 4/24, 8/24, 12/24, 1, 2, 3],
        'scale': 24
    }

    default_month = {
        'range':         [3, 4, 7, 12],
        'majorscale':    [1/2, 1, 2],
        'minorscale':    [1/6, 1/3, 1/2],
        'scale':         30
    }

    default_year = {
        'range':            [1, 2, 3, 5, 10, 15, 20, 30, 50, 100, 200, 400, 800],
        'major_scale':      [3/12, 4/12, 6/12, 1, 2, 3, 5, 5, 10, 20, 50, 100],
        'minor_scale':      [1/12, 1/12, 2/12, 3/12, 4/12, 6/12, 1, 1, 2, 5, 10, 20],
        'scale': 12
    }

    if 'my_second' in kwargs.keys():
        default_second.update(kwargs['my_second'])
    if 'my_minute' in kwargs.keys():
        default_second.update(kwargs['my_minute'])
    if 'my_hour' in kwargs.keys():
        default_second.update(kwargs['my_hour'])
    if 'my_day' in kwargs.keys():
        default_second.update(kwargs['my_day'])
    if 'my_month' in kwargs.keys():
        default_second.update(kwargs['my_month'])
    if 'my_year' in kwargs.keys():
        default_second.update(kwargs['my_year'])

    default_settings = {
        1:      {},
        2:      default_second,
        3:      default_minute,
        4:      default_hour,
        5:      default_day,
        6:      default_month,
        7:      default_year
    }

    tdelta = dt_stop - dt_start
    diff = tdelta.total_seconds()
    for ind in range(2, 8):
        range_ = default_settings[ind]['range']
        if (diff >= range_[0]) and (diff < range_[-1]):
            break
        else:
            if ind == 2:
                diff = diff/60
            elif ind ==3:
                diff = diff/60
            elif ind == 4:
                diff = diff/24
            elif ind == 5:
                diff = dttool.get_diff_months(dt_start, dt_stop)
            elif ind == 6:
                diff = dt_stop.Year - dt_start.Year

    setting = default_settings[ind]
    range_ = setting['range']
    for range_ind, value in enumerate(range_):
        if diff < value:
            break
    range_ind = range_ind - 1
    majorscale = setting['majorscale'][range_ind]
    minorscale = setting['minorscale'][range_ind]
    if majorscale < range_[0]:
        majorlocatorclass = eval(mdlocators[ind - 1])
        majorscale = majorscale * setting['scale']
    else:
        majorlocatorclass = eval(mdlocators[ind])
    if minorscale < range_[0]:
        minorlocatorclass = eval(mdlocators[ind - 1])
        minorscale = minorscale * setting['scale']
    else:
        minorlocatorclass = eval(mdlocators[ind])
    majorscale = int(majorscale)
    minorscale = int(minorscale)
    # for microseconds,
    if majorlocatorclass is mdates.MicrosecondLocator:
        interval = majorscale
        majorlocator = majorlocatorclass(interval=majorscale)
        if dt_start.minute != dt_stop.minute:
            fmt = "%M:%S.%f"
        else:
            fmt = "%S.%f"

        def formatter_microsecond(x, pos):
            dtx = mpl.dates.num2date(x)
            dtx = dtx.replace(tzinfo=None)
            delta = dtx - dt.datetime(dtx.year, dt.month, dtx.day, dtx.hour, dtx.minute, 0)
            if delta.total_seconds() == 0:
                fmt1 = "%M:%S.%f"
            else:
                fmt1 = "%S.%f"
            return dtx.strftime(fmt1)
        func_formatter = mpl.ticker.FuncFormatter(formatter_microsecond)
    if minorlocatorclass is mdates.MicrosecondLocator:
        interval = minorscale
        minorlocator = minorlocatorclass(interval=minorscale)

    # for second
    if majorlocatorclass is mdates.SecondLocator:
        by1 = range(0, 60, majorscale)
        majorlocator = majorlocatorclass(bysecond=by1, interval=1)
        if dt_start.hour != dt_stop.hour:
            fmt = "%H:%M:%S"
        else:
            fmt = "%M:%S"

        def formatter_second(x, pos):
            dtx = mpl.dates.num2date(x)
            dtx = dtx.replace(tzinfo=None)
            delta = dtx - dt.datetime(dtx.year, dtx.month, dtx.day, dtx.hour, 0, 0)
            if delta.total_seconds() == 0:
                fmt1 = "%H:%M:%S"
            else:
                fmt1 = "%M:%S"
            return dtx.strftime(fmt1)
        func_formatter = mpl.ticker.FuncFormatter(formatter_second)
    if minorlocatorclass is mdates.SecondLocator:
        by1 = range(0, 60, minorscale)
        minorlocator = minorlocatorclass(bysecond=by1, interval=1)

    # for minute
    if majorlocatorclass is mdates.MinuteLocator:
        by1 = range(0, 60, majorscale)
        majorlocator = majorlocatorclass(byminute=by1, interval=1)
        if dt_start.day != dt_stop.day:
            fmt = "%d %H:%M"
        else:
            fmt = "%H:%M"

        def formatter_minute(x, pos):
            dtx = mpl.dates.num2date(x)
            dtx = dtx.replace(tzinfo=None)
            delta = dtx - dt.datetime(dtx.year, dtx.month, dtx.day, 0, 0, 0)
            if delta.total_seconds() == 0:
                fmt1 = "%d/%m %H:%M"
            else:
                fmt1 = "%H:%M"
            return dtx.strftime(fmt1)
        func_formatter = mpl.ticker.FuncFormatter(formatter_minute)
    if minorlocatorclass is mdates.MinuteLocator:
        by1 = range(0, 60, minorscale)
        minorlocator = minorlocatorclass(byminute=by1, interval=1)

    # for hour
    if majorlocatorclass is mdates.HourLocator:
        by1 = range(0, 24, majorscale)
        majorlocator = majorlocatorclass(byhour=by1, interval=1)
        if dt_start.month != dt_stop.month:
            fmt = "%d/%m %H"
        elif dt_start.day != dt_stop.day:
            fmt = "%d %H:%M"
        else:
            fmt = "%H:%M"

        def formatter_hour(x, pos):
            dtx = mpl.dates.num2date(x)
            dtx = dtx.replace(tzinfo=None)
            delta = dtx - dt.datetime(dtx.year, dtx.month, dtx.day)
            if delta.total_seconds() == 0:
                fmt1 = "%b %d"
            else:
                fmt1 = "%H:%M"
            return dtx.strftime(fmt1)
        func_formatter = mpl.ticker.FuncFormatter(formatter_hour)
    if minorlocatorclass is mdates.HourLocator:
        by1 = range(0, 24, minorscale)
        minorlocator = minorlocatorclass(byhour=by1, interval=1)

    # for day
    if majorlocatorclass is mdates.DayLocator:
        temp = np.floor(31.5/majorscale)
        by1 = range(1, temp*majorscale, majorscale)
        majorlocator = majorlocatorclass(bymonthday=by1, interval=1)
        if dt_start.year != dt_stop.year:
            fmt = "%Y-%m-%d"
        else:
            fmt = "%b %d"

        def formatter_day(x, pos):
            dtx = mpl.dates.num2date(x)
            dtx = dtx.replace(tzinfo=None)
            delta = dtx - dt.datetime(dtx.year, 1, 1)
            if delta.total_seconds() == 0:
                fmt1 = "%Y-%m-%d"
            else:
                fmt1 = "%m-%d"
            return dtx.strftime(fmt1)
        func_formatter = mpl.ticker.FuncFormatter(formatter_day)
    if minorlocatorclass is mdates.DayLocator:
        temp = np.floor(31.5 / minorscale)
        by1 = range(1, temp * minorscale, minorscale)
        minorlocator = minorlocatorclass(bymonthday=by1, interval=1)

    # for month
    if majorlocatorclass is mdates.MonthLocator:
        by1 = range(1, 13, majorscale)
        majorlocator = majorlocatorclass(bymonth=by1, interval=1)
        fmt = "%Y-%m"

        def formatter_month(x, pos):
            dtx = mpl.dates.num2date(x)
            dtx = dtx.replace(tzinfo=None)
            delta = dtx - dt.datetime(dtx.year, 1, 1)
            if delta.total_seconds() == 0:
                fmt1 = "%Y-%m-%d"
            else:
                fmt1 = "%m-%d"
            return dtx.strftime(fmt1)
        func_formatter = mpl.ticker.FuncFormatter(formatter_month)
    if minorlocatorclass is mdates.MonthLocator:
        by1 = range(1, 13, minorscale)
        minorlocator = minorlocatorclass(bymonth=by1, interval=1)

    # for year
    if majorlocatorclass is mdates.YearLocator:
        majorlocator = majorlocatorclass(base=majorscale)

        def formatter_year(x, pos):
            dtx = mpl.dates.num2date(x)
            dtx = dtx.replace(tzinfo=None)
            fmt1 = "%Y"
            return dtx.strftime(fmt1)
        func_formatter = mpl.ticker.FuncFormatter(formatter_year)
    if minorlocatorclass is mdates.YearLocator:
        minorlocator = minorlocatorclass(base=minorscale)
    majorformatter = mdates.DateFormatter(fmt)
    majorformatter = func_formatter
    return majorlocator, minorlocator, majorformatter


# def timeline_format_function(x, pos=None):
#     x = mpl.dates.num2date(x)
#     if pos == 0:
#         # fmt = '%D %H:%M:%S.%f'
#         fmt = '%H:%M'
#     else:
#       fmt = '%H:%M'
#     label = x.strftime(fmt)
#     #label = label.rstrip("0")
#     #label = label.rstrip(".")
#     return label

#
# def set_ticks_datetime(fig, ax, axis='x', locator=None, formatter=None, interval=None, visable='on'):
#     xlim = getattr(ax, 'get_' + axis + 'lim')()
#     dxlim = np.diff(xlim) * 86400.
#     dsec = dxlim
#     dmin = dsec / 60.
#     dhour = dsec / 60.
#     dday = dhour / 24.
#     dmonth = dday / 30.
#     dyear = dday / 365.
#     if locator is None:
#         pass

#    locObj = getattr(mdates, locator)
    # majorlocator = mdates.MinuteLocator(interval=1)
    # formatter = mdates.AutoDateFormatter(dtlocator)
    # formatter.scaled[1 / (24. * 60.)] = matplotlib.ticker.FuncFormatter(my_format_function)
    # if iax < nax - 1:
    #     formatter = matplotlib.ticker.NullFormatter()
    # ax.xaxis.set_major_locator(dtlocator)
    # ax.xaxis.set_major_formatter(formatter)
    # ax.xaxis.set_minor_locator(mdates.SecondLocator(interval=10))
    # ax.xaxis.set_tick_params(labelsize='small')



# class CustomTicker(LogFormatterSciNotation):
#     # https://stackoverflow.com/questions/43923966/logformatter-tickmarks-scientific-format-limits
#     def __call__(self, x, pos=None):
#         if x not in [0.1,1,10]:
#             return LogFormatterSciNotation.__call__(self,x, pos=None)
#         else:
#             return "{x:g}".format(x=x)
# fig = plt.figure(figsize=[7,7])
# ax = fig.add_subplot(111)
# ax.set_yscale('log')
# ax.set_xscale('log')
# ax.plot(np.logspace(-4,4), np.logspace(-4,4))
#
# ax.xaxis.set_major_formatter(CustomTicker())
# plt.show()