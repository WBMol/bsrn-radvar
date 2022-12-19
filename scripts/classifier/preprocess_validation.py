import xarray
from datetime import datetime, timedelta
import os
import numpy as np
import pandas as pd

from general import settings as gsettings
from general import utils as gutils

# settings of raw input files
fdir_msg_base = os.path.join(gsettings.fdir_research_data, 'eumetsat', 'msgcpp')
fdir_in_msg = os.path.join(fdir_msg_base, 'raw', '{y}', '{m:02d}')
fmt_in_msg = 'in%Y%m%d%H%M00305SVMSG<ID>.nc'
fdir_in_class = os.path.join(fdir_msg_base, 'processed', 'fields', '{y}', '{m:02d}')
fname_in_class = 'classes_%Y%m%d.nc'

ids = {2014: '01UD', 2015: '01UD', 2016: 'E1UD'}
msg_radii = [5, 10, 15]


def preprocess_combined(year, msg_radius, overwrite=True):
    """
    Create a preprocessed validation dataset that only gives data if both CLAAS and Nubiscope are in agreement

    :param year: year to process
    :param int msg_radius: msg timeseries radius subset selection to use
    :param bool overwrite: whether to overwrite existing files
    :return:
    """
    date = datetime(year, 1, 1)
    date_stop = datetime(year+3, 1, 1)

    while date < date_stop:
        # check whether to process this date
        fpath_out = gutils.generate_processed_fpath(date, which='class_validation')
        if os.path.isfile(fpath_out) and not overwrite:
            print("Output file exists for date %s, not overwriting" % date)
            date += timedelta(days=1)
            continue

        # load the preprocessed CLAAS and Nubscope filepaths
        fpath_msg = gutils.generate_processed_fpath(date, which='msgcpp')
        fpath_nubi = gutils.generate_processed_fpath(date, which='nubiscope')

        if os.path.isfile(fpath_nubi) and os.path.isfile(fpath_msg):
            # load data
            d_msg = xarray.open_dataset(fpath_msg)
            d_nub = xarray.open_dataset(fpath_nubi)

            # interpolate both to a new time axis
            dtime_ax = pd.date_range(date, date + timedelta(minutes=59, hours=23), freq='min')
            d_msg = d_msg.interp(datetime=dtime_ax, method='nearest').sel(radius=msg_radius)
            d_nub = d_nub.reindex(datetime=dtime_ax, method='nearest')

            # get mutual nan mask
            nan_mask = d_msg.ccover.isnull() | d_nub.cldcover_total.isnull()

            # find mutual YES and NOs for clear-sky, mask the rest
            cs_msg = d_msg.ccover < 0.05
            cs_nub = d_nub.clearsky == 1
            cs = xarray.where(np.logical_and(cs_msg, cs_nub), 1, np.nan)
            cs = xarray.where(np.logical_and(~cs_msg, ~cs_nub), 0, cs)
            cs = xarray.where(np.logical_and(cs_msg, ~cs_nub), 2, cs)
            cs = xarray.where(np.logical_and(~cs_msg, cs_nub), 3, cs)
            cs = xarray.where(nan_mask, np.nan, cs)
            cs.attrs['description'] = '0=no cs, 1=cs, 2=msg-only, 3=nub-only'

            # find mutual YES and NOs for overcast
            oc_msg = ((d_msg.ccover > 0.95) & (d_msg.cot > 3.6)).values
            oc_nub = d_nub.overcast == 1
            oc = xarray.where(np.logical_and(oc_msg, oc_nub), 1, np.nan)
            oc = xarray.where(np.logical_and(~oc_msg, ~oc_nub), 0, oc)
            oc = xarray.where(np.logical_and(oc_msg, ~oc_nub), 2, oc)
            oc = xarray.where(np.logical_and(~oc_msg, oc_nub), 3, oc)
            oc = xarray.where(nan_mask, np.nan, oc)
            oc.attrs['description'] = '0=no oc, 1=oc, 2=msg-only, 3=nub-only'

            # combine to one data
            dset = xarray.Dataset(data_vars=dict(clearsky=cs, overcast=oc))
            dset.attrs['msg_radius_used'] = msg_radius
            if not os.path.exists(os.path.dirname(fpath_out)):
                os.makedirs(os.path.dirname(fpath_out))
            dset.to_netcdf(fpath_out)
        else:
            print("Not all input files exist for %s, skipping" % date)

        print("Processed validation set for %s" % date)
        date += timedelta(days=1)


if __name__ == "__main__":
    preprocess_combined(year=2014, msg_radius=10, overwrite=True)
