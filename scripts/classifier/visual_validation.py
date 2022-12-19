import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
from matplotlib import rc
from matplotlib.patches import Patch
import pandas as pd
import numpy as np
import os
import xarray
from datetime import datetime, timedelta
import copy
import matplotlib

import scripts.classifier.utils as utils
import scripts.classifier.settings as settings
import general.utils as gutils
import general.settings as gsettings
from utils import test_cases

matplotlib.use('Agg')
plt.style.use(gsettings.fpath_mplstyle)


def plot_classification(data, dt, msgcpp=None, nubi=None, plot_range='auto', res='1min', tmp_dir=None,
                        nubi_detail=False, add_wx=True):
    """
    visualize classification output for visual verification purposes

    :param data: preprocessed dataset
    :param msgcpp: cloud classification data array based on msgcpp, optional
    :param nubi: nubiscope data from Cabauw, optional
    :param datetime.datetime dt: datetime corresponding to the date of the data
    :param str, tuple plot_range: 'auto' or a tuple (hour_start, hour_stop)
    :param str res: dataset resolution (1min or 1sec)
    :param tmp_dir: optional temporary export dir used for exporting subsets for validation
    :param bool nubi_detail: whether to plot detailed nubiscope data or the simple version
    :param add_wx: whether to visualize the weather classifications or not
    :return:
    """
    # create figure
    fig = plt.figure(figsize=(8, 5), constrained_layout=True)
    ax1 = fig.add_subplot(111)

    # create subset
    plot_range = utils.get_time_plot_range(data.solar_elev, sea_limit=1) if plot_range == 'auto' else plot_range
    slicer = slice(dt + timedelta(hours=plot_range[0]), dt + timedelta(hours=plot_range[-1]))
    data = data.sel(time_rad=slicer, time_mcc=slicer, time_class=slicer)
    msgcpp = msgcpp.sel(radius=10) if msgcpp is not None else msgcpp
    c_nodata = (0.8, 0.8, 0.8)
    c_noclass = (0.95, 0.95, 0.95)

    # plot radiation data
    ldir, ldif, lcs = ('DHI', 'DIF', 'GHI$_{cs}$') if False else ('Direct (horizontal)', 'Diffuse', 'Clear-sky')
    lw = 1 if res == '1min' else 0.7
    ax1.plot(data.time_rad.values, data.ghi, color='black', linewidth=lw, label='Global', zorder=3)
    ax1.plot(data.time_mcc, data.ghi_cs, linewidth=lw, label=lcs, linestyle='--', color='darkslategray',
             alpha=0.8, zorder=1)
    ax1.plot(data.time_rad, data.dif, color='tab:blue', linewidth=lw, label=ldif, zorder=4, linestyle='-')
    ax1.plot(data.time_rad, data.dhi_sac, color='tab:orange', linewidth=lw, label=ldir, zorder=2, linestyle='-')
    ax1.plot(data.time_rad, np.zeros(data.time_rad.shape), color='black', alpha=0.8, lw=0.7, zorder=0)

    # set plot limits for the groups and irradiance data
    ymax = max(data.ghi.max(), data.ghi_cs.max())
    ymax = int(ymax + (100 - ymax % 100)) if not np.isnan(ymax) else 1000

    if msgcpp is None and nubi is None:
        nbars = 2
    elif msgcpp is None or nubi is None:
        nbars = 4
    else:
        nbars = 6
    if not add_wx:
        nbars -= 1
    extra_bar_size = nbars * 5

    bar_1_y0 = - ymax * extra_bar_size / 100
    bars_y0s = [bar_1_y0] + [bar_1_y0 / nbars * i for i in range(1, nbars)[::-1]]
    bar_padding = 10  # in percent of total bar height
    bars_offset = 2 / 100 * extra_bar_size / 100 * ymax  # in percent of total bar height
    y_label = []

    # plot classification results for both groups
    # group 2 (radiation classes)
    x = data.time_class.values
    y = [bars_y0s[-1] - bars_offset, bars_y0s[-1] * bar_padding / 100 - bars_offset]
    y_label.append((y[1] + y[0]) / 2)
    colors_g2 = [c_noclass, c_nodata, 'gray', plt.get_cmap('viridis_r')(0), gsettings.c_ce]
    cmap = mcolors.LinearSegmentedColormap.from_list('', colors_g2, N=len(colors_g2))
    Z = data.classes_group_2.values[:-1].reshape(1, len(data.classes_group_2) - 1)
    ax1.pcolormesh(x, y, Z, vmin=0, vmax=len(colors_g2), cmap=cmap, zorder=0, rasterized=True)

    # patch colour settings
    le = [Patch(facecolor=c, label=l) for (c, l) in zip(colors_g2, data.classes_group_2.class_names)]
    le_g2 = le[2:]

    props = dict(handletextpad=0.5, frameon=False, title_fontproperties=dict(weight='bold'), labelspacing=0.25)
    la_1 = ax1.legend(handles=le_g2, loc='upper left', bbox_to_anchor=(1., 1.04), **props, title='Irradiance')
    las = [la_1]

    # group 1 (weather classes)
    if add_wx:
        y = [yi + bars_y0s[-1] for yi in y]
        y_label.append((y[1] + y[0]) / 2)
        colors_g1 = [c_noclass, c_nodata, gsettings.c_clearsky, gsettings.c_overcast, gsettings.c_variable]
        cmap = mcolors.LinearSegmentedColormap.from_list('', colors_g1, N=len(colors_g1))
        Z = data.classes_group_1.values[:-1].reshape(1, len(data.classes_group_1) - 1)
        ax1.pcolormesh(x, y, Z, vmin=0, vmax=len(colors_g1), cmap=cmap, zorder=0, shading='auto', rasterized=True)

        le_res = le[:2]
        le_g1 = [Patch(facecolor=c, label=l) for (c, l) in zip(colors_g1, data.classes_group_1.class_names)][2:]

        la_2 = ax1.legend(handles=le_g1, loc='upper left', bbox_to_anchor=(1, 0.84), **props, title='Weather')
        la_3 = ax1.legend(handles=le_res, loc='upper left', bbox_to_anchor=(1, 0.64), **props, title='Residuals')
        las += [la_2, la_3]

    # group 3 & 4 (msgcpp)
    if msgcpp is not None:
        # build x axis such that it marks data point boundaries and is centered on retrieval time
        x = msgcpp.datetime
        x = x - pd.Timedelta(minutes=7.5)
        x = xarray.concat([x, x.isel(datetime=-1) + pd.Timedelta(minutes=15)], dim='datetime')

        # plot the cloud classes
        y = [yi + bars_y0s[-1] for yi in y]
        y_label.append((y[1] + y[0]) / 2)
        colors_g3 = [c_nodata, c_noclass] + [plt.get_cmap('tab20c_r')(i) for i in np.linspace(0, 1, 15)[6:]]
        cmap = mcolors.LinearSegmentedColormap.from_list('', colors_g3, N=len(colors_g3))
        Z = msgcpp.dom_class.values.reshape(1, len(msgcpp.dom_class))
        Z[np.isnan(Z)] = -1
        ax1.pcolormesh(x.values, y, Z, vmin=-1, vmax=10, cmap=cmap, zorder=0, rasterized=True)

        # add cloud class legend
        le_msg = [Patch(facecolor=c, label=l) for (c, l) in zip(colors_g3, msgcpp.dom_class.class_names)][2:]
        la_4 = ax1.legend(handles=le_msg, loc='upper left', bbox_to_anchor=(1, 0.49), **props,
                          title='Sat. Cloud type', ncol=3, columnspacing=1.)
        las.append(la_4)

        # plot cloud fraction
        y = [yi + bars_y0s[-1] for yi in y]
        y_label.append((y[1] + y[0]) / 2)
        cmap = copy.copy(plt.get_cmap('viridis_r'))
        cmap.set_under(c_nodata)
        Z = msgcpp.ccover.values.reshape(1, len(msgcpp.ccover))
        Z[np.isnan(Z)] = -1
        pcm_g4 = ax1.pcolormesh(x.values, y, Z, vmin=0, vmax=1, cmap=cmap, zorder=1, rasterized=True, )

    if nubi is not None:
        x = nubi.datetime
        x = x - pd.Timedelta(minutes=5)
        try:
            x = xarray.concat([x, x.isel(datetime=-1) + pd.Timedelta(minutes=10)], dim='datetime')

            # add cloud cover for low, middel and high clouds within one bar
            y = [yi + bars_y0s[-1] for yi in y]
            y_label.append((y[1] + y[0]) / 2)
            y_delta = y[1] - y[0]
            ylmh = [[y[0] + y_delta / 3 * (i - 1), y[0] + y_delta / 3 * i] for i in [1, 2, 3]]

            cmap = copy.copy(plt.get_cmap('viridis_r'))
            cmap.set_under(c_nodata)
            if nubi_detail:
                for i, cldcover in enumerate(['high', 'middle', 'low']):
                    Z = nubi['cldcover_%s' % cldcover].values.reshape(1, len(x) - 1)
                    Z[np.isnan(Z)] = -1
                    ax1.pcolormesh(x.values, ylmh[-i], Z, vmin=0, vmax=100, cmap=cmap, zorder=1, rasterized=True)
            else:
                Z = nubi['cldcover_total'].values.reshape(1, len(x) - 1)
                Z[np.isnan(Z)] = -1
                ax1.pcolormesh(x.values, y, Z, vmin=0, vmax=100, cmap=cmap, zorder=1, rasterized=True)

            # add classification type
            y = [yi + bars_y0s[-1] for yi in y]
            y_label.append((y[1] + y[0]) / 2)
            if nubi_detail:
                nct = xarray.where(nubi.obscuration_type == b'LF', 0, -1)
                flags = [b'LF', b'DF', b'HP', b'LC', b'TC', b'OC', b'BC', b'CI', b'CS']
                for flagnr, flag in enumerate(flags):
                    nct = xarray.where(nubi.obscuration_type == flag, flagnr + 1, nct)
                colors_g4 = [c_nodata, 'gray'] + [plt.get_cmap('tab20c_r')(i) for i in np.linspace(0, 1, 15)[6:]]
            else:
                nct = xarray.where(nubi.clearsky == 1, 1, 3)
                nct = xarray.where(nubi.overcast == 1, 2, nct)
                nct = xarray.where(nubi.cldcover_total.isnull(), 0, nct)
                colors_g4 = [c_nodata, gsettings.c_clearsky, gsettings.c_overcast, c_noclass]
                flags = ['clearsky', 'overcast', 'other']
            cmap = mcolors.LinearSegmentedColormap.from_list('', colors_g4, N=len(colors_g4))
            nct = nct.values.reshape(1, len(x) - 1)
            if nubi_detail:
                ax1.pcolormesh(x.values, y, nct, vmin=-1, vmax=9, cmap=cmap, zorder=1, rasterized=True)
                le_msg = [Patch(facecolor=c, label=l.decode()) for (c, l) in zip(colors_g4[2:], flags)]
                la_5 = ax1.legend(handles=le_msg, loc='upper left', bbox_to_anchor=(1, 0.3), **props,
                                  title='Nubi. Sky type', ncol=3, columnspacing=1.)
            else:
                ax1.pcolormesh(x.values, y, nct, vmin=0, vmax=3, cmap=cmap, zorder=1, rasterized=True)
                le_msg = [Patch(facecolor=c, label=l) for (c, l) in zip(colors_g4[1:], flags)]
                la_5 = ax1.legend(handles=le_msg, loc='upper left', bbox_to_anchor=(1, 0.3), **props,
                                  title='Nubi. Sky type', ncol=2, columnspacing=1.)

            las.append(la_5)
        except IndexError:
            pass

    # plot layout
    ax1.grid(alpha=0.2, linewidth=1.)
    ax1.set_ylabel('W m$^{-2}$')
    if fmt == 'pdf':
        ax1.set_xlabel('Time (UTC) | %s' % dt.strftime('%d %b %Y'))
    else:
        ax1.set_xlabel('Time (UTC) | %s | %s' % (dt.strftime('%d %b %Y'), res))
    ax1.legend(loc='lower center', ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.))
    ax1.set_xlim(slicer.start, slicer.stop)
    for la in las:
        ax1.add_artist(la)

    ax1.set_ylim(min(bars_y0s) - bars_offset, ymax)

    hour_range = plot_range[1] - plot_range[0]
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H' if hour_range > 10 else '%H:%M'))
    ax1.set_yticks(range(0, ymax + 1, 100))

    if msgcpp is not None:
        if nbars > 4:
            cax_g4 = fig.add_axes([1.01, 0.1 if nubi_detail else 0.12, 0.14, 0.02])
        else:
            cax_g4 = fig.add_axes([1.07, 0.2, 0.13, 0.02])
        cmap_g4 = fig.colorbar(pcm_g4, ticks=[0, 0.25, 0.5, 0.75, 1], cax=cax_g4, orientation='horizontal')
        cmap_g4.set_label('Cloud cover (%)', labelpad=-32, weight='bold')
        cmap_g4.set_ticklabels(['0', '25', '50', '75', '100'])

    # add labels to class bars
    if msgcpp is not None:
        dx = timedelta(hours=plot_range[0], minutes=2)
        tlha = 'left'
    else:
        dx = timedelta(hours=plot_range[1], minutes=2)
        tlha = 'left'
    tlc = 'white'
    fs = 8
    ax1.text(dt + dx, y_label[0], 'Irradiance', ha=tlha, va='center', color=tlc, fontsize=fs)
    if add_wx:
        ax1.text(dt + dx, y_label[1], 'Weather', ha=tlha, va='center', color=tlc, fontsize=fs)
    if msgcpp is not None:
        ax1.text(dt + dx, y_label[2 if add_wx else 1], 'Sat. cloud type', ha=tlha, va='center', color=tlc, fontsize=fs)
        ax1.text(dt + dx, y_label[3 if add_wx else 2], 'Sat. cloud cover', ha=tlha, va='center', color=tlc, fontsize=fs)
    if nubi is not None:
        ax1.text(dt + dx, y_label[-2], 'Nubi. cloud cover', ha=tlha, va='center', color=tlc, fontsize=fs)
        ax1.text(dt + dx, y_label[-1], 'Nubi. sky type', ha=tlha, va='center', color=tlc, fontsize=fs)

    # add data source label for presentation
    if False:
        ax1.text(dt + dx, 100, '$\\bf{Data\\ source}$:\nBSRN 1 Hz\nCabauw, NL\nKNMI', ha='left', va='center', color='black', fontsize=fs)

    # export and close
    if tmp_dir is None:
        fdir_out = os.path.join(settings.fdir_img, 'timeseries', res, dt.strftime('%Y'), dt.strftime('%m'))
    else:
        fdir_out = tmp_dir
    os.makedirs(fdir_out) if not os.path.exists(fdir_out) else None
    plt.savefig(os.path.join(fdir_out, '%s.%s' % (dt.strftime('%Y%m%d'), fmt)), dpi=gsettings.dpi, bbox_inches='tight')
    plt.close()


def visualise_range(dts, res='1sec', plot_range='auto', add_msg=False, add_nubi=False, tmp_dir=None, nubi_detail=False,
                    add_wx=True):
    """
    Batch generate time series classification quicklooks for a whole range of datetimes

    :param list dts: range of dates to process
    :param str res: source resolution of irradiance data
    :param tuple,str plot_range: specific plot range in the form of (hour start, hour end), or just 'auto'
    :param bool add_msg: whether to add msg
    :param bool add_nubi: whether to add nubiscope
    :param bool add_wx: whether to add weather classes at all
    :param tmp_dir: optional temporary export dir
    :return:
    """
    for dt in dts:
        # generate filepaths
        fpath_bsrn = gutils.generate_processed_fpath(dt, res, which='bsrn')
        fpath_msg = gutils.generate_processed_fpath(dt, res, which='msgcpp') if add_msg else None
        fpath_nubi = gutils.generate_processed_fpath(dt, res, which='nubiscope') if add_nubi else None

        # load data
        if os.path.isfile(fpath_bsrn):
            data_bsrn = gutils.load_timeseries_data(fpath_bsrn, apply_quality_control=True)
            if fpath_msg is not None and os.path.isfile(fpath_msg):
                data_msg = xarray.open_dataset(fpath_msg)
            else:
                if add_msg:
                    print('file does not exist: %s' % fpath_msg)
                data_msg = None
            if fpath_nubi is not None and os.path.isfile(fpath_nubi):
                data_nubi = xarray.open_dataset(fpath_nubi)
            else:
                if add_nubi:
                    print('file does not exist: %s' % fpath_nubi)
                data_nubi = None

            # call the plot function with data
            plot_classification(data_bsrn, dt, msgcpp=data_msg, nubi=data_nubi, plot_range=plot_range, res=res,
                                tmp_dir=tmp_dir, nubi_detail=nubi_detail, add_wx=add_wx)
        else:
            print("Skipped, file for %s does not exist" % dt)


def visualise_test_set(case_set, nubi_detail=True):
    fdir_test = os.path.join(settings.fdir_img, 'testset', case_set)
    visualise_range(dts=test_cases[case_set], plot_range='auto', add_msg=True, add_nubi=True, tmp_dir=fdir_test,
                    nubi_detail=nubi_detail)


def visualize_publication_set():
    dts = {
        'c1': (datetime(2016, 8, 15), (11, 12.5)),
        'c2': (datetime(2015, 4, 18), (4, 19)),
        'c3': (datetime(2015, 4, 3), (5, 18))
    }
    for dt, prange in dts.values():
        visualise_range([dt], res='1sec', plot_range=prange, add_msg=True, add_nubi=True, nubi_detail=False, add_wx=True,
                        tmp_dir=fdir_paper)


if __name__ == "__main__":
    # main setting
    paper_dir = True

    # automatic conditional settings
    fdir_paper = '../../paper-figures/images/paper_1'
    rc('font', size=9 if paper_dir else 8)
    fmt = 'pdf' if paper_dir else 'png'
    fdir_img = fdir_paper if paper_dir else settings.fdir_img

    # calling of plot scrips
    dts = gutils.generate_dt_range(datetime(2014, 1, 1), datetime(2016, 12, 31), delta_dt=timedelta(days=1))

    # visualise_range(dts, res='1sec', plot_range='auto', add_msg=True, add_nubi=True, nubi_detail=True, add_wx=True)
    # visualise_test_set(case_set='cs_to_oc')
