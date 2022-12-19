import matplotlib.pyplot as plt
import os
from matplotlib import rc
import xarray
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

import settings
from general import utils as gutils
from general import settings as gsettings

plt.style.use(gsettings.fpath_mplstyle)
rc('font', size=settings.fontsize)


def visualize_classification_seasonality(dts, cached=False, fmt='pdf'):
    """
    Visualize monthly classifiations for irradiance, satellite and nubiscope based observations

    :param dts: list of datetimes to build the validation figures for
    :param bool cached: whether to load the pre-calculated cached data to speed things up
    :param str fmt: output format
    :return:
    """
    # concatenate all claas2/nubiscope dates
    data = None
    tmp_file = './class_tmp_cached.nc'

    if cached and os.path.isfile(tmp_file):
        print("Loading data from cache")
        data = xarray.open_dataset(tmp_file)
    else:
        print("Generating dataset from individual files")
        for date in dts:
            # generate filepaths
            fpath_msg = gutils.generate_processed_fpath(date, which='class_validation')

            if os.path.isfile(fpath_msg):
                # load data
                d_msg = xarray.open_dataset(fpath_msg)
                data = xarray.concat([data, d_msg], dim='datetime') if data is not None else d_msg
            else:
                continue

        data.to_netcdf(tmp_file)

    # load statistics
    bsrn_stats = xarray.open_dataset(os.path.join(gsettings.fdir_bsrn_data, '1sec', 'statistics',
                                                  'daily_stats_bsrn_1sec.nc'))
    bsrn_stats = bsrn_stats.sel(date=slice(dts[0], dts[-1]))

    # clear-sky climatology
    cs_norm = xarray.where(data.clearsky.isnull(), 0, 1)
    cs_msg = xarray.where((data.clearsky == 1) | (data.clearsky == 2), 1, 0)
    cs_msg_monthly = (cs_msg / cs_norm).groupby('datetime.month').mean(dim='datetime') * 100

    cs_nub = xarray.where((data.clearsky == 1) | (data.clearsky == 3), 1, 0)
    cs_nub_monthly = (cs_nub / cs_norm).groupby('datetime.month').mean(dim='datetime') * 100

    cs_vld = xarray.where(data.clearsky == 1, 1, 0)
    cs_vld_monthly = (cs_vld / cs_norm).groupby('datetime.month').mean(dim='datetime') * 100

    cs_bsrn_monthly = (bsrn_stats['n_clearsky'] / bsrn_stats['n_possea']).groupby('date.month').mean(dim='date') * 100

    # overcast climatology
    oc_norm = xarray.where(data.overcast.isnull(), 0, 1)
    oc_msg = xarray.where((data.overcast == 1) | (data.overcast == 2), 1, 0)
    oc_msg_monthly = (oc_msg / oc_norm).groupby('datetime.month').mean(dim='datetime') * 100

    oc_nub = xarray.where((data.overcast == 1) | (data.overcast == 3), 1, 0)
    oc_nub_monthly = (oc_nub / oc_norm).groupby('datetime.month').mean(dim='datetime') * 100

    oc_vld = xarray.where(data.overcast == 1, 1, 0)
    oc_vld_monthly = (oc_vld / oc_norm).groupby('datetime.month').mean(dim='datetime') * 100

    oc_bsrn_monthly = (bsrn_stats['n_overcast'] / bsrn_stats['n_possea']).groupby('date.month').mean(dim='date') * 100

    # create figure
    fig, axes = plt.subplots(1, 2, figsize=gutils.get_image_size(ratio=0.5))

    # plot data
    x = cs_nub_monthly.month
    axes[0].plot(x, cs_msg_monthly, label='CLAAS2', marker='x')
    axes[0].plot(x, cs_nub_monthly, label='Nubiscope', marker='x')
    axes[0].plot(x, cs_vld_monthly, label='Validation', marker='x')
    axes[0].plot(x, cs_bsrn_monthly, label='BSRN', marker='x', color='tab:red')

    axes[1].plot(x, oc_msg_monthly, marker='x')
    axes[1].plot(x, oc_nub_monthly, marker='x')
    axes[1].plot(x, oc_vld_monthly, marker='x')
    axes[1].plot(x, oc_bsrn_monthly, marker='x', color='tab:red')

    # layout
    for ax in axes:
        ax.set_xlabel('Month of year')
        ax.set_xticks(range(1, 13), labels='JFMAMJJASOND')
        ax.set_xlim(1, 12)

    axes[0].set_ylabel('Clear-sky (% of time per day)')
    axes[1].set_ylabel('Overcast (% of time per day)')
    axes[0].set_ylim(0, 60)
    axes[1].set_ylim(0, 60)

    fig.legend(ncol=4, bbox_to_anchor=(0.5, 1.00), loc='center', frameon=False)

    for ax, label in zip(axes, 'ab'):
        ax.text(0., 1.01, '$\\bf{%s}$)' % label, transform=ax.transAxes, ha='left', va='bottom')

    # export and close
    plt.tight_layout()
    plt.savefig(os.path.join(settings.fdir_img_paper1, 'classification_climate.%s' % fmt), bbox_inches='tight')
    plt.close()


def visualize_cloud_fraction(dts, cached=False, fmt='pdf'):
    """

    :param dts:
    :param cached:
    :param fmt:
    :return:
    """
    # temporary file
    tmp_cf_file = './cache/cf_tmp_cached.nc'

    if cached and os.path.isfile(tmp_cf_file):
        cf = xarray.open_dataset(tmp_cf_file)
    else:
        # prepare output lists
        cf_nubs = None
        cf_msgs = None

        # prepare input data
        for date in dts:
            # load the preprocessed CLAAS and Nubscope filepaths
            fpath_msg = gutils.generate_processed_fpath(date, which='msgcpp')
            fpath_nubi = gutils.generate_processed_fpath(date, which='nubiscope')

            if os.path.isfile(fpath_nubi) and os.path.isfile(fpath_msg):
                # load data
                d_msg = xarray.open_dataset(fpath_msg)
                d_nub = xarray.open_dataset(fpath_nubi)

                # interpolate both to a new time axis
                dtime_ax = pd.date_range(date, date + timedelta(minutes=55, hours=23), freq='5min')
                d_msg = d_msg.interp(datetime=dtime_ax, method='nearest')
                d_nub = d_nub.reindex(datetime=dtime_ax, method='nearest')

                # get mutual nan mask
                nan_mask = d_msg.ccover.isnull() | d_nub.cldcover_total.isnull()

                cf_nub = xarray.where(nan_mask, np.nan, d_nub.cldcover_total / 100.)
                cf_msg = xarray.where(nan_mask, np.nan, d_msg.ccover)

                cf_nubs = xarray.concat([cf_nubs, cf_nub], dim='datetime') if cf_nubs is not None else cf_nub
                cf_msgs = xarray.concat([cf_msgs, cf_msg], dim='datetime') if cf_msgs is not None else cf_msg

        # create one data file
        cf = xarray.Dataset(data_vars=dict(msg=cf_msgs, nub=cf_nubs))
        cf = cf.dropna(dim='datetime')

        # export to temporary file
        cf.to_netcdf(tmp_cf_file)

    # create figure
    fig, ax = plt.subplots(1, 1, figsize=gutils.get_image_size(text_width=0.5, ratio=0.8))

    # plot data
    xbins = np.linspace(0, 1, 11)
    xbinc = (xbins[1:] + xbins[:-1]) / 2

    hist_nub, _ = np.histogram(cf.nub.sel(radius=5), bins=xbins, density=True)
    ax.bar(xbinc, hist_nub, width=0.07, label='Nubiscope', zorder=5, color='tab:blue', alpha=0.7)

    colors = plt.get_cmap('inferno')([0.4, 0.7, 0.9])
    for i, r in enumerate([5, 10, 15]):
        hist_msg, _ = np.histogram(cf.msg.sel(radius=r), bins=xbins, density=True)
        ax.scatter(xbinc, hist_msg, marker='x', color=colors[i], s=r*2, label='Satellite' if i == 0 else None,
                   zorder=6)

        # add corr label
        r2 = np.corrcoef(cf.nub.sel(radius=5), cf.msg.sel(radius=r))[0, 1] ** 2
        ax.text(0.01, 0.98 - i*0.08, 'r$_{%s}$$^2$ = %.2f' % (r, r2), ha='left', va='top', transform=ax.transAxes)

    # plot layout
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 8)
    ax.set_xlabel('Cloud fraction (-)')
    ax.set_ylabel('Probability density (-)')
    ax.legend(ncol=2, bbox_to_anchor=(0.5, 1.05), loc='center', frameon=False)
    ax.xaxis.set_ticks(xbins)

    # export and close
    plt.tight_layout()
    plt.savefig(os.path.join(settings.fdir_img_paper1, 'cloud_fraction_compare.%s' % fmt), bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    dts = gutils.generate_dt_range(datetime(2014, 1, 1), datetime(2016, 12, 31), delta_dt=timedelta(days=1))
    visualize_classification_seasonality(dts, cached=True, fmt='pdf')
    visualize_cloud_fraction(dts=dts, cached=True)
