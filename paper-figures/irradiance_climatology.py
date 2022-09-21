import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib import rc
import xarray
import os
import numpy as np
from datetime import datetime

from general import settings as gsettings
from general import utils as gutils
import settings

mpl.use('Agg')
plt.style.use(gsettings.fpath_mplstyle)
fontsize = 10
# mpl.rcParams['text.latex.preamble'] = r'\usepackage{sansmath} \sansmath'
# rc('text', usetex=False)
rc('font', size=fontsize)


def irradiance_climate(fmt='pdf', res='1sec'):
    """
    Figure about the total incoming irradiance climatology per month for all BSRN years of 1 minute data
    Direct/diffuse partitioning is compared to GHI, GHI is in turn compared to clear-sky values.

    :param str fmt: output format for image
    :param str res: input statistics file version (1sec or 1min)

    :return:
    """
    # load daily stats
    data = xarray.open_dataset(os.path.join(gsettings.fdir_research_data, 'Cabauw', 'BSRN', '1sec', 'statistics',
                                            'daily_stats_bsrn_%s.nc' % res))
    # data = data.where(data.date.dt.year == 2018, drop=True)

    # select only those with high enough quality
    data = data.where(data.fr_all > 0.95)

    # correct for missing data by taking the sum and dividing by fr_{var}, such that it is a mean instead of sum per day
    data['s_ghi'] = data['s_ghi'] / data['fr_all']
    data['s_dif'] = data['s_dif'] / data['fr_all']
    data['s_dni_sac'] = data['s_dni_sac'] / data['fr_all']

    # group data into years and months
    data_ = data.to_dataframe()
    data_ = data_.groupby([data_.index.year, data_.index.month]).mean().rename_axis(('year', 'month')).to_xarray()
    for v in ['ghi', 'ghi_cs', 'dif', 'dni_sac']:
        data_['s_%s' % v] = data_['s_%s' % v] / 86400

    # calculate the stats along requested axis
    data_mean = data_.mean(dim='year')
    data_std = data_.std(dim='year')

    # retrieve x-dim (all variables should share the same dim here)
    x = data_mean['month']
    dx = x[1] - x[0]
    img_h, img_v = gutils.get_image_size(text_width=1.)

    # create figure
    fig, ax = plt.subplots(1, 1, figsize=(img_h, img_v), constrained_layout=True)

    # plot data
    ax.bar(x, data_mean['s_ghi_cs'], zorder=3, color='white', width=0.8, linewidth=1.,
           label='GHI (CS)', edgecolor='gray')
    ax.bar(x - 0.2, data_mean['s_ghi'], label='GHI', width=0.39, linewidth=0, yerr=data_std['s_ghi'], capsize=3,
           error_kw=dict(linewidth=1), color=gsettings.c_ghi, edgecolor='black', zorder=5)

    ax.bar(x + 0.2, data_mean['s_dif'], label='DIF', width=0.39, linewidth=0, yerr=data_std['s_dif'], capsize=3,
           error_kw=dict(linewidth=1), color=gsettings.c_diff, edgecolor='black', zorder=5)
    ax.bar(x + 0.2, data_mean['s_dni_sac'], label='DNI', width=0.39, linewidth=0, yerr=data_std['s_dni_sac'],
           capsize=3, error_kw=dict(linewidth=1), bottom=data_mean['s_dif'], color=gsettings.c_dir, zorder=5)

    # add text labels
    dif_fr = data_mean['s_dif'] / data_mean['s_ghi'] * 100
    ghi_fr = data_mean['s_ghi'] / data_mean['s_ghi_cs'] * 100

    for i in range(len(dif_fr)):
        ax.text(x[i] + 0.2, 0, '%.0f' % dif_fr[i], va='bottom', ha='center', color='white', zorder=10)
        ax.text(x[i], data_mean['s_ghi_cs'][i] - 5, '%.0f' % ghi_fr[i], va='top', ha='center', color='black', zorder=10)

    # plot layout
    # ax.legend(loc='lower center', frameon=False, ncol=4, bbox_to_anchor=(.5, 1.01))
    ax.legend(loc='upper right', frameon=True, ncol=1)
    ax.set_xlim(x[0] - dx / 2, x[-1] + dx / 2)
    ax.set_ylim(0, 350.)
    ax.xaxis.set_ticks(settings.months)
    ax.xaxis.set_ticklabels(settings.months_labels)
    [ax.spines[spine].set_linewidth(0.) for spine in ['top', 'right']]
    ax.grid(axis='y', zorder=0, alpha=0.2, linewidth=.8)
    ax.grid(axis='x', alpha=0.)
    ax.set_ylabel('W m$^{-2}$')

    # export and close
    plt.savefig(os.path.join(settings.fdir_img_paper1, 'irradiance_climate_%s.%s' % (res, fmt)), bbox_inches='tight',
                dpi=gsettings.dpi * 1.5)
    plt.close()


def classification_histogram_overview(fmt='pdf'):
    """
    visualize the high level / simpler classification in a nice range of histograms (maybe box plots later)

    :param str fmt: export format
    :return:
    """
    # load daily data
    data = xarray.open_dataset(os.path.join(gsettings.fdir_bsrn_data, '1sec', 'statistics',
                                            'daily_stats_bsrn_1sec.nc'))
    data = data.where(data.fr_all > 0.95)
    vars_subset = ['n_overcast', 'n_clearsky', 'n_variable', 'n_shadow', 'n_sunshine', 'n_ce']

    # prepare the relative climatologies
    data_rel = (data[vars_subset] / data['n_possea']).to_dataframe()
    data_rel['n_sunshine'] = data_rel['n_sunshine'] + data_rel['n_ce']
    data_rel = data_rel.groupby([data_rel.index.year, data_rel.index.month]).mean()
    data_rel = data_rel.rename_axis(('year', 'month')).to_xarray() * 100

    # prepare absolute climatologies
    data_abs = data[vars_subset].to_dataframe()
    data_abs = data_abs.groupby(
        [data_abs.index.year, data_abs.index.month]).sum() / 3600 / 24  # TODO what about missing data?
    data_abs = data_abs.rename_axis(('year', 'month')).to_xarray()

    # prepare figure
    img_h, img_v = gutils.get_image_size()
    fig, axes = plt.subplots(2, 3, figsize=(img_h, img_v), constrained_layout=False)

    # plot the data
    error_kw = {'alpha': 0.5, 'capsize': 2}

    x = data_rel.month
    colors = [gsettings.c_shadow, gsettings.c_clearsky, gsettings.c_variable,
              gsettings.c_shadow, gsettings.c_sunshine, gsettings.c_ce]
    for ax, cat, c in zip(axes.flatten(), vars_subset, colors):
        ax.bar(x, data_rel[cat].mean(dim='year'), color=c, yerr=data_rel[cat].std(dim='year'), error_kw=error_kw,
               zorder=5)
        # ax.scatter(x, data_abs[cat].mean(dim='year'), color='black', zorder=5, s=2, marker='s')
        # ax_.plot(x, data_abs[cat].mean(dim='year'), color='black', alpha=0.5, zorder=6)
        ax.text(0.5, 1.01, cat.split('_')[-1], transform=ax.transAxes, ha='center', va='bottom')

    # add general layout
    for ax, label in zip(axes.flatten(), 'abcdef'):
        ax.grid(axis='y', linewidth=1., alpha=0.3, zorder=0)
        ax.set_xticks(settings.months)
        ax.set_xlim(0.5, 12.5)
        ax.xaxis.set_ticklabels([i[0] for i in settings.months_labels], rotation=0, fontsize=fontsize - 2)

        [ax.spines[spine].set_linewidth(0.) for spine in ['top', 'right']]
        ax.text(1., 1.01, '$\\bf{(%s)}$' % label, transform=ax.transAxes, ha='right', va='bottom')
        ax.set_ylim(0.)

    axes[1, 0].set_ylabel('Occurance (%)')
    axes[0, 0].set_ylabel('Occurance (%)')

    # manual axis limits
    for ax, limit in zip(axes.flatten(), [70, 10, 25, 100, 60, 30]):
        ax.set_ylim(0, limit)

    # export and close
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.25, hspace=0.25)
    plt.savefig(os.path.join(settings.fdir_img_paper1, 'class_climatology_overview_err.%s' % fmt), bbox_inches='tight',
                dpi=gsettings.dpi)
    plt.close()


def classification_histogram_overview_alt(fmt='pdf'):
    """
    visualize the high level / simpler classification in a nice range of histograms (maybe box plots later)

    :param str fmt: export format
    :return:
    """
    # load daily data
    data = xarray.open_dataset(os.path.join(gsettings.fdir_bsrn_data, '1sec', 'statistics', 'daily_stats_bsrn_1sec.nc'))
    data = data.where(data.fr_all > 0.95)
    vars_subset = ['n_overcast', 'n_clearsky', 'n_variable', 'n_shadow', 'n_sunshine', 'n_ce', 'n_possea']

    # prepare the relative climatologies
    data_rel = (data[vars_subset] / data['n_possea']).to_dataframe()
    data_rel['n_sunshine'] = data_rel['n_sunshine'] + data_rel['n_ce']
    data_rel = data_rel.groupby([data_rel.index.year, data_rel.index.month]).mean()
    data_rel = data_rel.rename_axis(('year', 'month')).to_xarray() * 100

    # prepare absolute climatologies
    data_abs = data[vars_subset].to_dataframe()
    data_abs['n_sunshine'] += data_abs['n_ce']
    data_abs = data_abs.groupby([data_abs.index.year, data_abs.index.month]).mean() / 3600
    data_abs = data_abs.rename_axis(('year', 'month')).to_xarray()

    # prepare figure
    img_h, img_v = gutils.get_image_size()
    fig, axes = plt.subplots(2, 3, figsize=(img_h, img_v), constrained_layout=False)

    # define plot settings and axes
    ws = 0.55
    x = data_rel.month
    shift = 0.10
    x1 = x - shift
    x2 = x + shift
    # c_shadow = gsettings.c_shadow
    c_shadow = 'darkgray'

    for ax, d in zip(axes, [data_rel, data_abs]):
        # a, d
        ax[0].bar(x1, d['n_overcast'].mean(dim='year'), color=gsettings.c_overcast, zorder=5, label='overcast',
                  width=0.55, yerr=d['n_overcast'].std(dim='year'), error_kw=dict(ecolor='white'))
        ax[0].bar(x, d['n_shadow'].mean(dim='year'), color=c_shadow, zorder=4, label='shadow', width=0.8,
                  yerr=d['n_shadow'].std(dim='year'), error_kw=dict(ecolor=gsettings.c_overcast))

        # b, e
        ax[1].bar(x, d['n_sunshine'].mean(dim='year'), color=gsettings.c_sunshine, zorder=4, label='sunshine',
                  width=0.8, yerr=d['n_sunshine'].std(dim='year'), error_kw=dict(alpha=0.6))
        ax[1].bar(x1, d['n_clearsky'].mean(dim='year'), color=gsettings.c_clearsky, zorder=5, label='clear-sky',
                  width=0.6, yerr=d['n_clearsky'].std(dim='year'), error_kw=dict(ecolor='black', alpha=0.8))

        # c, f
        ax[2].bar(x1, d['n_variable'].mean(dim='year'), color='turquoise', zorder=5, label='variable',
                  width=ws, yerr=d['n_variable'].std(dim='year'), error_kw=dict(capsize=0, ecolor='teal'))
        ax[2].bar(x2, d['n_ce'].mean(dim='year'), yerr=d['n_ce'].std(dim='year'), color=gsettings.c_ce, zorder=4,
                  label='cloud-enh.', width=ws, error_kw=dict(capsize=0, ecolor='darkred'), alpha=1)
        # ax[2].errorbar(x1, d['n_variable'].mean(dim='year'), yerr=d['n_variable'].std(dim='year'), capsize=2,
        #                ecolor='tab:green', fmt='s', label='variable', markersize=3, markerfacecolor='tab:green',
        #                mec='tab:green')
        # ax[2].errorbar(x2, d['n_ce'].mean(dim='year'), yerr=d['n_ce'].std(dim='year'), capsize=2,
        #                ecolor='tab:red', fmt='s', label='cloud enh.', markersize=3, markerfacecolor='tab:red',
        #                mec='tab:red')

    kwargs = dict(handlelength=1.3, handletextpad=0.3, columnspacing=.5, frameon=False, ncol=2, loc='lower center',
                  bbox_to_anchor=(0.5, 1.1))
    for ax in axes[0]:
        ax.legend(**kwargs)

    # add general layout
    for ax, label in zip(axes.flatten(), 'abcdef'):
        ax.grid(axis='y', linewidth=1., alpha=0.3, zorder=0)
        ax.set_xticks(settings.months)
        ax.set_xlim(0.5 - shift, 12.5 + shift)
        ax.xaxis.set_ticklabels([i[0] for i in settings.months_labels], rotation=0, fontsize=fontsize - 2)

        [ax.spines[spine].set_linewidth(0.) for spine in ['top', 'right']]
        ax.text(0., 1.01, '$\\bf{%s}$)' % label, transform=ax.transAxes, ha='left', va='bottom')

    axes[0, 0].set_ylabel('Occurrence (%)')
    axes[1, 0].set_ylabel('Hours per day')

    # manual axis limits
    for ax, limit in zip(axes.flatten(), [100, 61, 30, 12, 10, 5]):
        ax.set_ylim(0, limit)
        if limit == 12:
            ax.set_yticks(range(0, limit + 1, 3))
        if limit == 5:
            ax.set_yticks(range(0, limit + 1, 1))
        if limit == 50:
            ax.set_yticks(range(0, limit + 1, 10))
        if limit == 25:
            ax.set_yticks(range(0, limit + 1, 5))

    # export and close
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.25, hspace=0.3)
    plt.savefig(os.path.join(settings.fdir_img_paper1, 'class_climatology_overview.%s' % fmt), bbox_inches='tight',
                dpi=gsettings.dpi)
    plt.close()


if __name__ == "__main__":
    # irradiance_climate(fmt='pdf', res='1sec')
    # cloud_enhancement_distribution(fmt='png', distribution='density')
    # shadow_distribution(fmt='png', distribution='density')
    # classification_histogram_overview(fmt='png')
    classification_histogram_overview_alt(fmt='pdf')
    # classification_histogram_overview_second(fmt='png')
    # hod_toy_distribution(variable='cmf', fmt='png')
