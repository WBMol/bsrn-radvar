import time
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta as datedelta

from scripts.processor import settings


def split_dtrange_into_ranges(time_start, time_stop, ranges):
    """
    given an input starting and stopping datetime, split the range into separate days

    returns list of ranges
    :param datetime.datetime time_start:
    :param datetime.datetime time_stop:
    :param str ranges: 'day' of 'month'
    """
    delta_dt = time_stop - time_start
    assert ranges in ['days', 'months'], "ranges parameter should be 'day' or 'month', not '%s'" % ranges
    assert delta_dt.total_seconds() > 0, "Stop should be later than start datetime!"

    if ranges == 'days':
        if delta_dt.total_seconds() < 86400:
            return [(time_start, time_stop)]
        else:
            # the first range might be an incomplete day
            time_start_ = time_start
            time_stop_ = (time_start + timedelta(days=1)).replace(hour=0, minute=0, second=0)
            ranges = [(time_start_, time_stop_)]

            # calculate to total days yet to process
            total_days = delta_dt.total_seconds() / 86400 - 1

            # the next part will be all full days, done in a loop
            time_start = ranges[0][1]

            for i in range(0, int(total_days)):
                time_start_ = time_start + timedelta(days=i)
                time_stop_ = time_start + timedelta(days=i + 1)
                ranges.append((time_start_, time_stop_))

            # the last range might again be incomplete, or does not exist
            if time_stop != time_start + timedelta(days=int(total_days)):
                ranges.append((time_start + timedelta(days=int(total_days)), time_stop))

            return ranges
    elif ranges == 'months':
        # get month number since year 0 (to handle year transitions)
        month_start = time_start.year * 12 + time_start.month
        month_stop = time_stop.year * 12 + time_stop.month

        if month_stop - month_start == 0:
            return [(time_start, time_stop)]
        else:
            # the first range might be an incomplete month
            time_start_ = time_start
            time_stop_ = datetime(time_start.year + (time_start.month // 12), ((time_start.month % 12) + 1), 1)
            ranges = [(time_start_, time_stop_)]

            # calculate total months yet to process
            total_months = month_stop - month_start - 1

            # the next part will be all full days, done in a loop
            time_start = ranges[0][1]

            for i in range(0, int(total_months)):
                time_start_ = time_start + datedelta(months=i)
                time_stop_ = time_start + datedelta(months=i + 1)
                ranges.append((time_start_, time_stop_))

            # we end with a potentially ending datetime again
            if time_stop != time_start + datedelta(months=int(total_months)):
                ranges.append((time_start + datedelta(months=int(total_months)), time_stop))

            return ranges


def timeit(f):
    if settings.debug:
        def wrapped(*args, **kw):

            ts = time.time()
            result = f(*args, **kw)
            te = time.time()

            print('func %r took %2.4f sec' % (f.__name__, te - ts))
            return result
    else:
        def wrapped(*args, **kw):
            return f(*args, **kw)

    return wrapped
