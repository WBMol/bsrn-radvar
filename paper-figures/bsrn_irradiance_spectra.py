import xarray
from scipy import fft
from datetime import datetime
import matplotlib.pyplot as plt
import os
import numpy as np
import sys
from matplotlib import rc

sys.path.append('./')

import general.utils as gutils
import general.settings as gsettings
import scripts.processor.utils as putils
import settings

plt.style.use(gsettings.fpath_mplstyle)
fdir_cache = './paper-figures/cache/'
rc('font', size=settings.fontsize)


@putils.timeit
def prepare_input(dates, cache=True, overwrite=False, source='bsrn_1sec'):
    """
    Load daily timeseries data into one big timeseries

    :param dates: list of datetime objects to load data for
    :param cache: whether to use caching (saves processing time)
    :param overwrite: whether to overwrite existing cached data
    :return:
    """
    fpath_cache = os.path.join(fdir_cache, 'input_%s.npy' % source)

    if (cache and os.path.isfile(fpath_cache)) and not overwrite:
        print('loading input from cache')
        input = np.load(fpath_cache)
    else:
        print('processing input from scratch')
        input = None
        for date in dates:
            # prepare filepath and load data
            if source == 'bsrn_1sec':
                fpath = gutils.generate_processed_fpath(date, res='1sec')
                data_ = gutils.load_timeseries_data(fpath)
                input_ = data_.ghi
                input_ = input_.interpolate_na(method='linear', dim='time_rad')
                input_ = xarray.where(data_.solar_elev < -5, 0, input_)

                # prepare input
                input = input_ if input is None else xarray.concat([input, input_], dim='time_rad')

            elif source == 'frost_10hz':
                fdir = os.path.join(gsettings.fdir_research_data, 'Fieldwork', 'fesstval', 'slocs-sensors', 'public', 'level2')
                fname = 'fval_wur_radgrid10Hz_l2_rsds_v00_%s.nc' % date.strftime('%Y%m%d')
                fpath = os.path.join(fdir, fname)
                data_ = xarray.open_dataset(fpath).sel(sensor_id=11)
                input_ = data_.rsds.where(data_.quality == 1)
                input_ = input_.interpolate_na(method='linear', dim='datetime')

                # prepare input
                input = input_ if input is None else xarray.concat([input, input_], dim='datetime')

            # close data
            data_.close()

        # extract values and cache
        print("%i values, or %.2f%% of data is missing" % (int(np.sum(~np.isfinite(input))),
                                                           np.mean(~np.isfinite(input)) * 100))
        input = xarray.where(~np.isfinite(input), 0, input)
        input = input.values
        np.save(fpath_cache, input) if cache else None
        print('generated input array')
    return input


@putils.timeit
def run_fft(input, threads=1, res='1sec'):
    """
    Run the fft on the input
    """
    n = len(input)

    # run the fft
    y = fft.rfft(input, workers=threads)[:n//2] / n
    y = np.abs(y)**2 * 2

    # generate x axis
    d = {'1sec': 1, '1min': 60, '10hz': 0.1}[res]
    x = fft.fftfreq(n, d=d)[:n//2]

    # check spectra
    print('Variance test: %.4f%% similar' % (np.var(input) / sum(y[1:]) * 100))

    return (x, y)


@putils.timeit
def bin_fft(fft, cache, overwrite, res='1sec'):
    """
    Make a logarithmic binning of spectra to reduce noise

    :param fft: raw fft output
    :param cache: whether to use caching (saves processing time)
    :param overwrite: whether to overwrite existing cached data
    :param res: resolution of input data (1sec, 1min, 10hz)
    :return:
    """
    fpath_cache = os.path.join(fdir_cache, 'fft_binned_%s.npy' % res)

    if cache and not overwrite and os.path.isfile(fpath_cache):
        (new_x, new_y) = np.load(fpath_cache)
    else:
        x, y = fft
        spectrum = xarray.DataArray(data=y, dims=('f',), coords=dict(f=x))
        bins = np.logspace(-7, 1.5, 101)

        new_y = spectrum.groupby_bins(group='f', bins=bins).mean().values
        new_x = (bins[1:] + bins[:-1]) / 2.

        if cache:
            np.save(fpath_cache, (new_x, new_y))

    return new_x, new_y


@putils.timeit
def visualize_fft(fft_1sec, fft_1min=None, fft_10hz=None, fmt='png', style='loglog', cache=True, overwrite=False):
    """
    visualize the input and output

    :param fft_1sec: fourier transform of BSRN timeseries at native 1 Hz
    :param fft_1min: optional fourier transform of BSRN timeseries at 1 min resampled resolution
    :param fft_10hz: optional fourier transform of FROST / FESSTVaL timeseries at 10 Hz resolution
    :param str fmt: image output format
    :param str style: plotting style: 'loglog' or 'variance'
    :param bool cache: whether to use caching (saves processing time)
    :param bool overwrite: whether to overwrite existing cached data
    :return:
    """
    # create figure, axes
    _, ax = plt.subplots(1, 1, figsize=gutils.get_image_size(text_width=0.8, ratio=0.7))

    if style == 'variance':
        x, y = fft_1sec
        y = y * x
        fft_1sec = (x, y)

    # smooth the fft
    fft_ = bin_fft(fft_1sec, cache=cache, overwrite=overwrite, res='1sec')
    fft_1min_ = bin_fft(fft_1min, cache=cache, overwrite=overwrite, res='1min') if fft_1min is not None else None
    fft_10hz_ = bin_fft(fft_10hz, cache=cache, overwrite=overwrite, res='10hz') if fft_10hz is not None else None

    # plot smoothed fft
    ax.plot(fft_[0], fft_[1], alpha=1, label='BSRN 1 sec', color='tab:red', zorder=6, lw=2)
    if fft_1min is not None:
        ax.plot(fft_1min_[0], fft_1min_[1], alpha=1, label='BSRN 1 min', color='tab:blue', ls='--', zorder=7, lw=2)
    if fft_10hz is not None:
        ax.plot(fft_10hz_[0], fft_10hz_[1], alpha=0.5, label='FESSTVaL 10 Hz', color='black', ls='-', zorder=8)

    # set axes to log
    ax.set_xscale('log')
    ax.set_yscale('log') if style == 'loglog' else None
    ax.set_xlim(1e-6, 1) if style == 'loglog' else ax.set_xlim(1e-7, 0.5)

    # add power law scaling
    if style == 'loglog':
        xpow = np.logspace(-4, -0.5, 10)
        n = np.argmin(np.abs(fft_[0] - 1e-2))
        c = fft_[1][n] / np.power(1e-2, -5/3) * 50
        ypow = c * np.power(xpow, -5/3)
        ax.plot(xpow, ypow, linestyle='--', color='black', zorder=6, label='$f^{-5/3}$', alpha=0.7)
    elif style == 'variance':
        ax.set_ylim(0, 1e-4)

    # add supporting labels
    ylim = ax.get_ylim()
    xlim = ax.get_xlim()
    for freq, label in zip([1/600, 1/86400, 1/43200, 1/60, 1/3600, 1/10, 0.5, 1, 5],
                           ['10 min', '24 h', '12 h', '1 min', '1 h', '10 s', '2 s', '1 s', '0.2 s']):
        if freq >= xlim[0] and freq <= xlim[1]:
            ax.plot([freq] * 2, ylim, color='black', ls='-', lw=1, alpha=0.1, zorder=0)
            ax.text(freq, ylim[0] * 5, label, va='bottom', ha='right', rotation=90, alpha=0.7, zorder=7)
    ax.set_ylim(*ylim)

    # plot styling
    ax.set_xlabel('f (s$^{-1}$)')
    if style == 'loglog':
        ax.set_ylabel('PSD (W$^2$ m$^{-4}$ s)')
    elif style == 'variance':
        ax.set_ylabel('$f$ * PSD (W$^2$ m$^{-4}$)')
    ax.legend(ncol=4, bbox_to_anchor=(0.5, 1.01), frameon=False, loc='lower center')
    ax.grid(False, axis='x')

    # export and close
    plt.tight_layout()
    plt.savefig(os.path.join(settings.fdir_img_paper1, 'ghi_fft_%s.%s' % (style, fmt)), bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    # settings
    cache = True
    overwrite = False
    bsrn = True
    frost = True

    # generate input
    if bsrn:
        # set the temporal range of BSRN input data
        ts = datetime(2016, 1, 1)
        te = datetime(2017, 1, 1)
        dts = gutils.generate_dt_range(ts, te)

        # prepare the data
        x = prepare_input(dates=dts, cache=cache, overwrite=overwrite, source='bsrn_1sec')
        x_1min = np.mean(x.reshape(-1, 60), axis=1)

        # run fft
        xfft_1sec = run_fft(x, threads=4, res='1sec')
        xfft_1min = run_fft(x_1min, threads=4, res='1min')
    else:
        xfft_1sec = None
        xfft_1min = None
    
    if frost:
        # set the temporal range of FROST input data
        ts = datetime(2021, 6, 14)
        te = datetime(2021, 6, 30)
        dts = gutils.generate_dt_range(ts, te)

        # prepare the data
        x = prepare_input(dates=dts, cache=cache, overwrite=overwrite, source='frost_10hz')

        # run the fft
        xfft_10hz = run_fft(x, threads=4, res='10hz')
    else:
        xfft_10hz = None


    # visualize
    visualize_fft(fft_1sec=xfft_1sec, fft_1min=xfft_1min, fft_10hz=xfft_10hz, fmt='pdf', style='loglog', cache=cache, overwrite=overwrite)
