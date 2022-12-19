import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib as mpl
from matplotlib import rc
import xarray
import os
import numpy as np

from general import settings as gsettings
from general import utils as gutils
import settings

mpl.use('Agg')
rc('font', size=settings.fontsize)

plt.style.use(gsettings.fpath_mplstyle)

fdir_cache = './cache/'
os.makedirs(fdir_cache) if not os.path.exists(fdir_cache) else None


def prepare_distribution_data(variable, cth_threshold, season):
    """
    Load data and return the raw segment data for a range of arguments

    :param variable:
    :param cth_threshold:
    :param season:
    :return:
    """
    segments = xarray.open_dataset(
        os.path.join(gsettings.fdir_bsrn_data, '1sec', 'statistics', '1sec_segments_total_stats_all_%s.nc' % variable))
    # os.path.join(gsettings.fdir_bsrn_data, '1sec_segments_total_stats_all_%s_20220321_1958.nc' % variable))
    subset = ['duration', 'u200', 'dir_min', 'solar_angle', 'start_time', 'ghi_mean_rel']
    subset.append('ce_max') if variable == 'cloud-enh' else None
    subset.append('cth') if cth_threshold is not None else None
    segments = segments[subset]
    segments = segments.where(segments.solar_angle > 5).dropna(dim='segment')

    if season == 'summer':
        segments = segments.where((segments.start_time.dt.month >= 5) & (segments.start_time.dt.month <= 8))
    elif season == 'winter':
        segments = segments.where((segments.start_time.dt.month >= 11) | (segments.start_time.dt.month <= 2))

    if cth_threshold is not None:
        segments = segments.where(segments.cth < cth_threshold * 1e3).dropna(dim='segment')

    return segments


def segment_size_distribution(fmt='pdf', variable='shadow', cth_threshold=None, add_boxplots=False, season='all'):
    """
    Generate the shadow distribution figure

    :param str fmt: png or pdf, export format
    :param str variable: 'shadow' or 'ce'
    :param cth_threshold: optional filter of cloud top height (km) to support argumentation that most of it is BL clouds
    :param add_boxplots: whether to add the boxplots to extract some typical sizes, defaults to yes
    :param str season: an optional season subset. Only summers summer (may jun jul aug) or winter (nov dec jan feb)
    :return:
    """
    # load and prep data
    segments = prepare_distribution_data(variable=variable, cth_threshold=cth_threshold, season=season)

    # create fig, ax
    img_h, img_v = gutils.get_image_size(ratio=0.5)
    fig, axes = plt.subplots(1, 2, figsize=(img_h, img_v), constrained_layout=True)

    # calculate histogram
    bin_edges = np.concatenate([np.arange(0, 10, 1), np.logspace(1, 6, 101)])
    bin_edges = np.unique(bin_edges.astype(np.int64))
    handles = []
    counts = []

    if variable == 'shadow':
        thresholds = [120][::-1]
        colors = [plt.get_cmap('viridis')(i) for i in np.linspace(0., 0.8, len(thresholds))][::-1]
    else:
        thresholds = [1, 100, 200]
        colors = [plt.get_cmap('inferno')(i) for i in np.linspace(0., 0.8, len(thresholds))]
        # colors = [plt.get_cmap('inferno')(0.6)]

    bp_data = {'duration': [], 'size': []}

    for i, threshold in enumerate(thresholds):
        if variable == 'shadow':
            ss = segments.where((segments.dir_min <= threshold), drop=False)
        else:
            ss = segments.where((segments.ce_max >= threshold), drop=False)

        ss = ss.dropna(dim='segment')
        ss['size'] = ss['duration'] * ss['u200']

        for ax, var in zip(axes, ['duration', 'size']):
            binned_data = ss[[var]].groupby_bins(var, bin_edges, include_lowest=False)
            y_mean = binned_data.count()[var]

            dx = np.diff(bin_edges)
            x = bin_edges[:-1] + dx / 2
            y_mean = y_mean / dx / y_mean.sum()

            # plot results
            l, = ax.plot(x, y_mean, alpha=1., zorder=5, color=colors[i], lw=1.5)
            handles.append(l) if var == 'duration' else None

            if add_boxplots:
                bp_data[var].append(ss[var].values)

            if var == 'duration':
                counts.append(len(ss[var]))

            if i == 0:
                alpha = - 1.66
                if var == 'size':
                    kx = np.logspace(2.3, 5.3 if variable == 'shadow' else 4.5, num=11)
                    sigma = 7e1
                else:
                    kx = np.logspace(1.5, 5. if variable == 'shadow' else 4.2, num=11)
                    sigma = 2e1
                ky = sigma * np.power(kx, alpha)
                ax.plot(kx, ky, linestyle='--', color='black', alpha=0.5, zorder=6)
                ax.text(kx[-2], ky[-2], '$x^{-1.66}$', ha='left', va='bottom', color='black', alpha=0.5)

            # plot labels of distribution means
            # idxmax = np.nanargmax(y_mean.values)
            # xmax = x[idxmax]
            unit = 's' if var == 'duration' else 'm'
            print('%s \t threshold=%s:\t mean = %i, median = %i (%s)' % (var[:4], threshold,
                                                                         int(ss[var].mean()), int(ss[var].median()),
                                                                         unit))

            # if add_boxplots:
            #     if variable == 'shadow':
            #         ax.text(0.95, 0.85 + i * 0.05, '%i %s' % (xmax, unit), transform=ax.transAxes, color=colors[i],
            #                 ha='right', va='top')
            #     else:
            #         ax.text(0.95, 0.85 + i * 0.05, '%i %s' % (xmax, unit), transform=ax.transAxes, color=colors[i],
            #                 ha='right', va='top')
            # else:
            #     if variable == 'shadow':
            #         ax.text(0.05, 0.05 + i * 0.05, '%i %s' % (xmax, unit), transform=ax.transAxes, color=colors[i],
            #                 ha='left', va='bottom')
            #     else:
            #         ax.text(0.05, 0.15 - i * 0.05, '%i %s' % (xmax, unit), transform=ax.transAxes, color=colors[i],
            #                 ha='left', va='bottom')

            # export the distribution data
            np.save(os.path.join(fdir_cache, 'x_%s.npy' % var), x)
            np.save(os.path.join(fdir_cache, 'y_%s_%s_%s.npy' % (var, threshold, season)), y_mean)

    if add_boxplots:
        for ax, var in zip(axes, ['duration', 'size']):
            ax_ = ax.twinx()
            ax_.set_ylim(-1.5, 20)
            boxes = ax_.boxplot(bp_data[var], positions=range(len(thresholds)), vert=False, whis=[5, 95], showfliers=False,
                                showmeans=True, widths=0.75,
                                meanprops=dict(marker='s', markerfacecolor='black', markeredgewidth=0))
            ax_.set_yticks([])
            for b in ['boxes', 'medians']:
                for i, j in enumerate(boxes[b]):
                    j.set_color(colors[i])
            for b in ['caps', 'whiskers']:
                for i, j in enumerate(boxes[b]):
                    j.set_color(colors[i // 2])
            for b in ['means']:
                for i, j in enumerate(boxes[b]):
                    j.set_markerfacecolor(colors[i])

    # subplot-specific layout
    for ax in axes:
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(1e0, 4e5)
        ax.set_xticks([1, 1e1, 1e2, 1e3, 1e4, 1e5])
        ax.xaxis.set_minor_locator(mticker.LogLocator(base=10, subs=np.arange(0.1, 1., 0.1), numticks=100))
        ax.xaxis.set_minor_formatter(mticker.NullFormatter())

    label = 'Shadow' if variable == 'shadow' else 'Cloud enhancement'
    axes[0].set_xlabel('%s duration (s)' % label)
    axes[0].set_ylabel('Probability density (s$^{-1}$)')
    axes[0].set_ylim(1e-9, 1e-1)

    axes[1].set_xlabel('%s length (m)' % label)
    axes[1].set_ylabel('Probability density (m$^{-1}$)')
    axes[1].set_ylim(1e-9, 1e-1)

    # labeling
    if variable == 'shadow':
        # labels = ['min(DNI) $\\leq$ %s Wm$^{-2}$' % i for i in thresholds]
        labels = ['min(DNI) $\\leq$ %s W m$^{-2}$\nn = %i' % i for i in zip(thresholds, counts)]
    else:
        # labels = ['max(CE) $\\geq$ %s Wm$^{-2}$' % i for i in thresholds]
        labels = ['max(CE) $\\geq$ %s W m$^{-2}$\nn = %i' % i for i in zip(thresholds, counts)]
    l = fig.legend(handles=handles, labels=labels, frameon=False, ncol=3, bbox_to_anchor=(0.5, 1.01),
                   loc='lower center', handletextpad=0.5, columnspacing=2, handlelength=1.5)
    for ax, label in zip(axes, 'ab'):
        ax.text(0., 1.01, '$\\bf{%s}$)' % label, transform=ax.transAxes, ha='left', va='bottom')

    # save and export
    fname = 'segment_size_%s_pdf.%s' % (variable, fmt)
    fname = fname.replace('_pdf', '_low_clouds_pdf') if cth_threshold is not None else fname
    fname = fname.replace('_pdf', '_%s_pdf' % season) if season != 'all' else fname
    plt.savefig(os.path.join(settings.fdir_img_paper1, fname), bbox_inches='tight',
                dpi=gsettings.dpi)
    plt.close()


def segment_size_distribution_merged(fmt='pdf', add_boxplots=False):
    """
    Generate the shadow & CE distribution figure

    :param str fmt: png or pdf, export format
    :param add_boxplots: whether to add the boxplots to extract some typical sizes, defaults to yes
    :return:
    """
    # load and prep data
    segments_sh = prepare_distribution_data(variable='shadow', cth_threshold=None, season='all')
    segments_ce = prepare_distribution_data(variable='cloud-enh', cth_threshold=None, season='all')

    # create fig, ax
    img_h, img_v = gutils.get_image_size(ratio=1)
    fig, axes_ = plt.subplots(2, 2, figsize=(img_h, img_v), constrained_layout=False)

    # calculate histogram
    bin_edges = np.concatenate([np.arange(0, 10, 1), np.logspace(1, 6, 101)])
    bin_edges = np.unique(bin_edges.astype(np.int64))

    for variable, segments, axes in zip(['shadow', 'cloud-enh'], [segments_sh, segments_ce], axes_):
        handles = []
        counts = []

        # segments = segments.isel(segment=slice(0, 10000))

        if variable == 'shadow':
            thresholds = [4, 60, 120][::-1]
            colors = [plt.get_cmap('viridis')(i) for i in np.linspace(0., 0.8, len(thresholds))][::-1]
        else:
            thresholds = [1, 100, 200]
            colors = [plt.get_cmap('inferno')(i) for i in np.linspace(0., 0.8, len(thresholds))]

        bp_data = {'duration': [], 'size': []}

        for i, threshold in enumerate(thresholds):
            if variable == 'shadow':
                ss = segments.where((segments.dir_min <= threshold), drop=False)
            else:
                ss = segments.where((segments.ce_max >= threshold), drop=False)

            ss = ss.dropna(dim='segment')
            ss['size'] = ss['duration'] * ss['u200']

            for ax, var in zip(axes, ['duration', 'size']):
                binned_data = ss[[var]].groupby_bins(var, bin_edges, include_lowest=False)
                y_mean = binned_data.count()[var]

                dx = np.diff(bin_edges)
                x = bin_edges[:-1] + dx / 2
                y_mean = y_mean / dx / y_mean.sum()

                # plot results
                l, = ax.plot(x, y_mean, alpha=1., zorder=5, color=colors[i], lw=1.5)
                handles.append(l) if var == 'duration' else None

                if add_boxplots:
                    bp_data[var].append(ss[var].values)

                if var == 'duration':
                    counts.append(len(ss[var]))

                if i == 0:
                    alpha = -1.66
                    if var == 'size':
                        kx = np.logspace(2.3, 5.3 if variable == 'shadow' else 4.5, num=11)
                        sigma = 7e1
                    else:
                        kx = np.logspace(1.5, 5. if variable == 'shadow' else 4.2, num=11)
                        sigma = 2e1
                    ky = sigma * np.power(kx, alpha)
                    ax.plot(kx, ky, linestyle='--', color='black', alpha=0.5, zorder=6)
                    ax.text(kx[-2], ky[-2], '$x^{-1.66}$', ha='left', va='bottom', color='black', alpha=0.5)

                # plot labels of distribution means
                unit = 's' if var == 'duration' else 'm'
                print('%s \t threshold=%s:\t mean = %i, median = %i (%s)' % (var[:4], threshold,
                                                                             int(ss[var].mean()), int(ss[var].median()),
                                                                             unit))

        if add_boxplots:
            for ax, var in zip(axes, ['duration', 'size']):
                ax_ = ax.twinx()
                ax_.set_ylim(-1.5, 20)
                boxes = ax_.boxplot(bp_data[var], positions=[0, 1, 2], vert=False, whis=[5, 95], showfliers=False,
                                    showmeans=True, widths=0.75,
                                    meanprops=dict(marker='s', markerfacecolor='black', markeredgewidth=0))
                ax_.set_yticks([])
                for b in ['boxes', 'medians']:
                    for i, j in enumerate(boxes[b]):
                        j.set_color(colors[i])
                for b in ['caps', 'whiskers']:
                    for i, j in enumerate(boxes[b]):
                        j.set_color(colors[i // 2])
                for b in ['means']:
                    for i, j in enumerate(boxes[b]):
                        j.set_markerfacecolor(colors[i])

        # subplot-specific layout
        for ax in axes:
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlim(1e0, 4e5)
            ax.set_xticks([1, 1e1, 1e2, 1e3, 1e4, 1e5])
            ax.xaxis.set_minor_locator(mticker.LogLocator(base=10, subs=np.arange(0.1, 1., 0.1), numticks=100))
            ax.xaxis.set_minor_formatter(mticker.NullFormatter())

        label = 'Shadow' if variable == 'shadow' else 'Cloud enhancement'
        axes[0].set_xlabel('%s duration (s)' % label)
        axes[0].set_ylabel('Probability density (s$^{-1}$)')
        axes[0].set_ylim(1e-9, 1e-1)

        axes[1].set_xlabel('%s size (m)' % label)
        axes[1].set_ylabel('Probability density (m$^{-1}$)')
        axes[1].set_ylim(1e-9, 1e-1)

        # labeling
        if variable == 'shadow':
            # labels = ['min(DNI) $\\leq$ %s Wm$^{-2}$' % i for i in thresholds]
            labels = ['min(DNI) $\\leq$ %s W m$^{-2}$\nn = %i' % i for i in zip(thresholds, counts)]
            # labels = ['%s W m$^{-2}$\nn = %i' % i for i in zip(thresholds, counts)]
        else:
            # labels = ['max(CE) $\\geq$ %s Wm$^{-2}$' % i for i in thresholds]
            labels = ['max(CE) $\\geq$ %s W m$^{-2}$\nn = %i' % i for i in zip(thresholds, counts)]
        l = axes[0].legend(handles=handles, labels=labels, frameon=False, ncol=3, bbox_to_anchor=(0.11, 1.02),
                           loc='lower left', handletextpad=0.5, columnspacing=1.7, handlelength=2)
    for ax, label in zip(axes_.flatten(), 'abcd'):
        ax.text(0., 1.01, '$\\bf{%s}$)' % label, transform=ax.transAxes, ha='left', va='bottom')

    # save and export
    fname = 'segment_size_merged_pdf.%s' % fmt
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.4, hspace=0.6)
    # fig.set_constrained_layout_pads(hspace=0.2)
    plt.savefig(os.path.join(settings.fdir_img_paper2, fname), bbox_inches='tight', dpi=gsettings.dpi)
    plt.close()


def correct_domain_size(date, l, u=None):
    doy = date.dt.dayofyear
    d = np.cos((doy + 11 + 365/2) / 365 * (2 * np.pi)) * 17.5e3 + 37.5e3

    d = d if u is None else d * u
    return d * l / (d - l - 1)


def segment_size_distribution_extended(fmt='pdf', add_boxplots=False, season='all'):
    """
    Generate the shadow distribution figure

    :param str fmt: png or pdf, export format
    :param add_boxplots: whether to add the boxplots to extract some typical sizes, defaults to yes
    :param str season: an optional season subset. Only summers summer (may jun jul aug) or winter (nov dec jan feb)
    :return:
    """
    # load and prep data
    segments = prepare_distribution_data(variable='shadow', cth_threshold=None, season=season)

    # create fig, ax
    img_h, img_v = gutils.get_image_size(ratio=0.5)
    fig, axes = plt.subplots(1, 2, figsize=(img_h, img_v), constrained_layout=True)

    # calculate histogram
    bin_edges = np.concatenate([np.arange(0, 10, 1), np.logspace(1, 6, 101)])
    bin_edges = np.unique(bin_edges.astype(np.int64))
    handles = []
    counts = []

    thresholds = [120][::-1]
    colors = [plt.get_cmap('viridis')(i) for i in np.linspace(0., 0.8, len(thresholds))][::-1]

    bp_data = {'duration': [], 'size': []}

    for i, threshold in enumerate(thresholds):
        ss = segments.where((segments.dir_min <= threshold), drop=False)
        ss = ss.dropna(dim='segment')
        ss['duration_corr'] = correct_domain_size(ss['start_time'], ss['duration'])
        ss['size'] = ss['duration'] * ss['u200']
        ss['size_corr'] = correct_domain_size(ss['start_time'], ss['size'], ss['u200'])
        ss['area'] = np.pi * (ss['size'] / 2) ** 2

        for ax, var in zip(axes, ['duration', 'size']):
            dx = np.diff(bin_edges)
            x = bin_edges[:-1] + dx / 2

            # calculate distribution
            binned_data = ss[[var]].groupby_bins(var, bin_edges, include_lowest=False)
            y_mean = binned_data.count()[var]
            y_mean = y_mean / dx / y_mean.sum()

            # plot results
            l, = ax.plot(x, y_mean, alpha=1., zorder=5, color=colors[i], lw=1.5)
            handles.append(l) if var == 'duration' else None

            # calculate also the corrected distribution
            binned_data = ss[[var + '_corr']].groupby_bins(var + '_corr', bin_edges, include_lowest=False)
            y_mean = binned_data.count()[var + '_corr']
            y_mean = y_mean / dx / y_mean.sum()

            # plot results
            l, = ax.plot(x, y_mean, alpha=1., zorder=5, color=colors[i], lw=1.5, linestyle=':')

            if var == 'size':
                bin_edges = np.logspace(1, 9, 101)
                bin_edges = np.unique(bin_edges.astype(np.int64))
                dx = np.diff(bin_edges)
                x = bin_edges[:-1] + dx / 2
                # calculate also the corrected distribution
                binned_data = ss[['area']].groupby_bins('area', bin_edges, include_lowest=False)
                y_mean = binned_data.count()['area']
                y_mean = y_mean / dx / y_mean.sum()

                # plot results
                l, = ax.plot(x, y_mean, alpha=1., zorder=5, color=colors[i], lw=1.5, linestyle='--')

            if add_boxplots:
                bp_data[var].append(ss[var].values)

            if var == 'duration':
                counts.append(len(ss[var]))

            if i == 0:
                alpha = -5 / 3
                if var == 'size':
                    kx = np.logspace(2.3, 5.3, num=11)
                    sigma = 7e1
                else:
                    kx = np.logspace(1.5, 5., num=11)
                    sigma = 2e1

                ky = sigma * np.power(kx, alpha)
                ax.plot(kx, ky, linestyle='--', color='black', alpha=0.5, zorder=6)
                ax.text(kx[-2], ky[-2], '$x^{-5/3}$', ha='left', va='bottom', color='black', alpha=0.5)
                #
                # if var == 'size':
                #     ky = sigma * np.power(kx, -1.87)
                #     axes[2].plot(kx, ky, linestyle='--', color='tab:red', alpha=0.5, zorder=6)
                #     ax.text(kx[-2], ky[-2], '$x^{-1.87}$', ha='left', va='bottom', color='black', alpha=0.5)

            # plot labels of distribution means
            unit = 's' if var == 'duration' else 'm'
            print('%s \t threshold=%s:\t mean = %i, median = %i (%s)' % (var[:4], threshold,
                                                                         int(ss[var].mean()), int(ss[var].median()),
                                                                         unit))

    # subplot-specific layout
    for ax in axes:
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(1e0, 4e5)
        ax.set_xticks([1, 1e1, 1e2, 1e3, 1e4, 1e5])
        ax.xaxis.set_minor_locator(mticker.LogLocator(base=10, subs=np.arange(0.1, 1., 0.1), numticks=100))
        ax.xaxis.set_minor_formatter(mticker.NullFormatter())

    # axes[2].set_xlim(1e1, 1e9)
    # axes[2].set_xticks([1, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9])

    label = 'Shadow'
    axes[0].set_xlabel('%s duration (s)' % label)
    axes[0].set_ylabel('Probability density (s$^{-1}$)')
    axes[0].set_ylim(1e-9, 1e-1)

    axes[1].set_xlabel('%s length (m)' % label)
    axes[1].set_ylabel('Probability density (m$^{-1}$)')
    axes[1].set_ylim(1e-9, 1e-1)

    # labeling
    labels = ['min(DNI) $\\leq$ %s W m$^{-2}$\nn = %i' % i for i in zip(thresholds, counts)]
    l = fig.legend(handles=handles, labels=labels, frameon=False, ncol=3, bbox_to_anchor=(0.5, 1.01),
                   loc='lower center', handletextpad=0.5, columnspacing=2, handlelength=1.5)
    for ax, label in zip(axes, 'abc'):
        ax.text(0., 1.01, '$\\bf{%s}$)' % label, transform=ax.transAxes, ha='left', va='bottom')

    # save and export
    fname = 'segment_size_shadow_corrected_pdf.%s' % fmt
    fname = fname.replace('_pdf', '_%s_pdf' % season) if season != 'all' else fname
    plt.savefig(os.path.join(settings.fdir_img_paper2, fname), bbox_inches='tight',
                dpi=gsettings.dpi)
    plt.close()


if __name__ == "__main__":
    # segment_size_distribution(fmt='png', variable='shadow', cth_threshold=None, season='winter')
    # segment_size_distribution(fmt='pdf', variable='cloud-enh', cth_threshold=None, add_boxplots=True)
    segment_size_distribution_merged(fmt='pdf', add_boxplots=True)
    # segment_size_distribution_extended(fmt='png')
