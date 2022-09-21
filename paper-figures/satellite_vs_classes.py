import xarray
import os
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np

from general import utils as gutils
from general import settings as gsettings
import settings

plt.style.use(gsettings.fpath_mplstyle)
fdir_stat_file = os.path.join(gsettings.fdir_bsrn_data, '1sec', 'statistics')
rc('font', size=settings.fontsize)


def load_cloudclass_climatology(density=True, radius=5):
    """
    Load preprocessed satellite cloud classification data

    :param density:
    :param radius:
    :return:
    """
    # file dir and name settings - run the prepare_cloudclass_climatology in segment_test_figures
    fdir = '../scripts/conditional_stats/output/'
    fname = 'msgcpp_class_climate_reference_%s_r%i.npy' % ('density' if density else 'counts', radius)

    # cloud class bins x-axis edges
    x_edge = np.arange(0.5, 10.5, 1)
    return np.load(os.path.join(fdir, fname)), x_edge


def ce_vs_cloudclass(fmt='png', radius=10, density=True, weighted=False):
    """
    Barplot of dominant cloud classes for CE vs. climatology

    :param str fmt: image output format
    :param radius: size of radius around Cabauw
    :param bool density: normalize the classes to combine to 100%
    :return:
    """
    # cloud enhancement thresholds
    ce_b, ce_c = 1.20, 1.30
    colors = ['tab:green', 'tab:orange', 'tab:blue']
    alphas = [0.5, 0.75, 1.]
    x = np.arange(1, 10)
    radii = [5, 10, 15] if radius == 'all' else [radius]
    offset = [-0.25, 0, 0.25] if radius == 'all' else [0]
    width = 0.2 if radius == 'all' else 0.95
    bar_size = 200 if len(radii) == 1 else 15

    # load segments data
    data = xarray.open_dataset(os.path.join(fdir_stat_file, '1sec_segments_total_stats_all_cloud-enh.nc'))

    # drop the segments that have no cloud information
    data = data.where(~np.isnan(data.cloud_class.sel(radius=radii[0])), drop=True)

    # setup figure
    fig, axes = plt.subplots(1, 3, figsize=gutils.get_image_size(ratio=0.35), constrained_layout=True)

    # MSGCPP, reference climate
    for i, r in enumerate(radii):
        hist_a, x_edge = load_cloudclass_climatology(density=density, radius=r)
        bars = axes[0].bar(x + offset[i], hist_a, linewidth=0., zorder=5, width=width)

        for j, bar in enumerate(bars):
            bar.set_color(colors[j // 3])
            bar.set_alpha(alphas[j % 3])

    # CE weak
    data_ = data.where(data.ghi_max_rel <= ce_b)
    axes[1].text(0.05, 0.95, 'n = %i' % np.sum(data.ghi_max_rel <= ce_b), transform=axes[1].transAxes,
                 ha='left', va='top', alpha=0.8)
    for i, r in enumerate(radii):
        w_b = data_.duration if weighted else None
        hist_b, _ = np.histogram(data_.cloud_class.sel(radius=r), bins=x_edge, density=density, weights=w_b)
        bars = axes[1].bar(x + offset[i], hist_b, linewidth=0., zorder=5, width=width)
        for j, bar in enumerate(bars):
            bar.set_color(colors[j//3])
            bar.set_alpha(alphas[j % 3])

        if i == 0 and radius != 'all':
            subset = data_.sel(radius=r)
            dur_b = subset.duration.groupby_bins(subset.cloud_class, bins=x_edge)
            for j, dur in enumerate(dur_b.mean()):
                va = 'bottom' if hist_b[j] < 0.5 else 'top'
                voffset = 0 if va == 'bottom' else -0.008
                axes[1].text(x[j], hist_b[j] + voffset, '%.1f' % (dur/60), ha='center', fontsize=8, va=va, zorder=5)

    # CE strong
    data_ = data.where(data.ghi_max_rel > ce_c)
    axes[2].text(0.05, 0.95, 'n = %i' % np.sum(data.ghi_max_rel > ce_c), transform=axes[2].transAxes,
                 ha='left', va='top', alpha=0.8)
    for i, r in enumerate(radii):
        w_c = data_.duration if weighted else None
        hist_c, _ = np.histogram(data_.cloud_class.sel(radius=r), bins=x_edge, density=density, weights=w_c)
        bars = axes[2].bar(x + offset[i], hist_c, linewidth=0., zorder=5, width=width)
        for j, bar in enumerate(bars):
            bar.set_color(colors[j//3])
            bar.set_alpha(alphas[j % 3])

        if i == 0 and radius != 'all':
            subset = data_.sel(radius=r)
            dur_b = subset.duration.groupby_bins(subset.cloud_class, bins=x_edge)
            for j, dur in enumerate(dur_b.mean()):
                axes[2].text(x[j], hist_c[j], '%.1f' % (dur/60), ha='center', fontsize=8, va='bottom')

    # plot layout
    for ax, label in zip(axes, 'abc'):
        # ax.set_xlabel('Cloud classification')
        ax.text(0., 1.01, '$\\bf{%s}$)' % label, transform=ax.transAxes, ha='left', va='bottom')
        ax.xaxis.set_ticks(x)
        ax.set_xlim(0.4, 9.6)
        ax.xaxis.set_ticklabels(["Ci", "Cs", "Cb", "Ac", "As", "Ns", "Cu", "Sc", "St"], fontsize=8)
        ax.grid(axis='x', alpha=0)
        ylim = 0.61 if radius == 'all' else 0.6
        if label == 'a':
            ax.set_ylabel('Fraction of time')
        elif not weighted:
            ax.set_ylabel('Fraction of events')
        ax.set_ylim(0, ylim)
        ax.set_yticks(np.arange(0, ylim + 0.01, 0.1))

    axes[0].text(0.5, 1.01, 'Satellite Climatology', transform=axes[0].transAxes, ha='center', va='bottom')
    axes[1].text(0.5, 1.01, 'max(CE) $\\leq$ %s GHI$_{cs}$' % ce_b, transform=axes[1].transAxes, ha='center',
                 va='bottom')
    axes[2].text(0.5, 1.01, 'max(CE) $>$ %s GHI$_{cs}$' % ce_c, transform=axes[2].transAxes, ha='center',
                 va='bottom')

    # export and close
    fname_out = 'ce_vs_cloudclass_r%s_%s.%s' % (radius, 'time' if weighted else 'counts', fmt)
    plt.savefig(os.path.join(settings.fdir_img_paper2, fname_out), bbox_inches='tight', dpi=gsettings.dpi)
    plt.close()


def cloud_types_per_shadow_size(fmt='png'):
    """
    Create a grouped bin distribution of cloud types per shadow chord length

    :param str fmt: output image format
    :return:
    """
    # load segment data
    data = xarray.open_dataset(os.path.join(gsettings.fdir_bsrn_data, '1sec', 'statistics',
                                            '1sec_segments_total_stats_all_shadow.nc'))
    data = data.sel(radius=10)

    # subset the part where we have msgcpp
    data = data.where(np.isfinite(data.cot))
    data = data.dropna(dim='segment')

    # calculate the size
    data['size'] = data['duration'] * data['u200']

    # group the shadow sizes according to certain bins
    cds_bins = [0, 1000, 2000, 5000, 10000, 20000, 100000]
    groups = data.groupby_bins(group='size', bins=cds_bins)

    # build the normalized histograms
    hists = []
    bins = []
    for (bin, group) in sorted(groups):
        hist = np.histogram(group.cloud_class, bins=np.arange(0.5, 10.5, 1), density=True, weights=group.duration)
        hists.append(hist[0])
        bins.append((bin.left, bin.right))
    hists = np.array(hists)

    # create fig, ax
    fig, ax = plt.subplots(1, 1, figsize=gutils.get_image_size(text_width=0.75))
    cs = [plt.get_cmap('tab20c_r')(i) for i in np.linspace(0, 1, 15)[6:]]
    labels = ["Ci", "Cs", "Cb", "Ac", "As", "Ns", "Cu", "Sc", "St"]

    # plot per cloud type the contribution for each shadow size
    for i in range(len(hists[0])):
        y = hists[:, i]

        if i == 0:
            ax.bar(range(len(cds_bins)-1), y, color=cs[i], label=labels[i])
        else:
            y_bot = hists[:, :i].sum(axis=1)
            ax.bar(range(len(cds_bins)-1), y, bottom=y_bot, color=cs[i], label=labels[i])

    # plot layout
    ax.legend(ncol=1, bbox_to_anchor=(1.1, 0.5), loc='center', frameon=False)
    ax.set_ylim(0, 1)
    ax.xaxis.set_ticks(range(len(cds_bins)-1), labels=['(%i, %i]' % (i//1e3, j//1e3) for (i, j) in bins])
    ax.set_xlabel('Shadow chord length bins (km)')
    ax.set_ylabel('Normalized cloud type occurrence (-)')

    # plot export
    plt.tight_layout()
    plt.savefig(os.path.join(settings.fdir_img_paper2, 'cloud_types_per_shadow_size.%s' % fmt), bbox_inches='tight',
                dpi=gsettings.dpi)
    plt.close()


if __name__ == "__main__":
    ce_vs_cloudclass(fmt='pdf', radius='all', weighted=True)
    # cloud_types_per_shadow_size(fmt='pdf')
