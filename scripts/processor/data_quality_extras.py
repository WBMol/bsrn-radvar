import xarray
from datetime import datetime
import os
import numpy as np

import utils
import settings


def prepare_quality_flags_from_official_1min(dt_start, dt_stop):
    """
    Extract quality flags from official 1 minute BSRN data from PANGAEA that were preprocessed to
    daily files

    :param dt_start:
    :param dt_stop:
    :return:
    """
    day_ranges = utils.split_dtrange_into_ranges(dt_start, dt_stop, ranges='days')

    for (ts, _) in day_ranges:
        # create fpath to raw monthly file
        fdir_in = settings.fdir_out.format(res='1min', y=ts.year, m=ts.month)
        fname = ts.strftime('%Y%m%d.nc')
        fpath = os.path.join(fdir_in, fname)
        data = xarray.open_dataset(fpath)

        # calculate quality flags (0=good, 1=invalid/missing)
        q_ghi = xarray.where(np.isnan(data.ghi), 1, 0).astype(np.uint8)
        q_dif = xarray.where(np.isnan(data.dif), 1, 0).astype(np.uint8)
        q_dir = xarray.where(np.isnan(data.dni), 1, 0).astype(np.uint8)
        q_all = xarray.where((q_ghi + q_dif + q_dir) > 0, 1, 0).astype(np.uint8)
        quality = xarray.Dataset(data_vars=dict(ghi=q_ghi, dif=q_dif, dir=q_dir, all=q_all))

        # create new output dir/path
        fpath_out = fpath.replace('processed', 'quality')
        if not os.path.exists(os.path.dirname(fpath_out)):
            os.makedirs(os.path.dirname(fpath_out))

        # export the quality flags per date
        quality.to_netcdf(fpath_out)
        print('Extracted quality flags from %s' % ts.strftime('%Y%m%d.nc'))


def compare_quality_flags(dt_start, dt_stop):
    """
    Calculate how much the quality flags are similar

    :param dt_start:
    :param dt_stop:
    :return:
    """
    day_ranges = utils.split_dtrange_into_ranges(time_start, time_stop, ranges='days')
    v1, v2 = 'custom_sw_quality', 'official_sw_quality'

    t_true = []
    t_fa = []
    t_miss = []

    for (ts, _) in day_ranges:
        # create fpath to raw monthly file
        fdir_in = settings.fdir_out.format(res='1sec', y=ts.year, m=ts.month)
        data = xarray.open_dataset(os.path.join(fdir_in, ts.strftime('%Y%m%d.nc')))
        data = data[[v1, v2]].where(data.solar_elev > 0).dropna(dim='time_rad')

        pct_agree = (data[v1] == data[v2]).mean().values * 100
        pct_pbias = ((data[v1] == 1) & (data[v2] == 0)).mean().values * 100
        pct_nbias = ((data[v1] == 0) & (data[v2] == 1)).mean().values * 100

        print('[%s] Agree: %.2f%%, false alarm: %.2f%%, missed: %.2f%%' % (ts.strftime('%Y%m%d'),
                                                                           pct_agree, pct_pbias,
                                                                           pct_nbias))

        t_true.append(pct_agree)
        t_fa.append(pct_pbias)
        t_miss.append(pct_nbias)

    print('[%s till %s] Agree: %.2f%%, false alarm: %.2f%%, missed: %.2f%%' % (dt_start.strftime('%Y%m%d'),
                                                                               dt_stop.strftime('%Y%m%d'),
                                                                               np.mean(t_true), np.mean(t_fa),
                                                                               np.mean(t_miss)))


if __name__ == "__main__":
    # settings
    time_start = datetime(2016, 1, 1)
    time_stop = datetime(2017, 1, 1)

    # generate flags
    prepare_quality_flags_from_official_1min(dt_start=time_start, dt_stop=time_stop)

    # compare flags
    compare_quality_flags(dt_start=time_start, dt_stop=time_stop)
