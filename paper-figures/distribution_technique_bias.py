import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
import os

from general import settings as gsettings
from general import utils as gutils
import settings

plt.style.use(gsettings.fpath_mplstyle)
rc('font', size=settings.fontsize)

# settings of distribution
bin_edge = np.logspace(2.2, 5, 32)
bins = 0.5 * (bin_edge[1:] + bin_edge[:-1])
fdir_distr = '../../csd/data/'
plc = (0.2, 0.2, 0.2)  # power law color (label)


def f(arr):
    return np.ma.masked_where(arr == 0, arr)


def make_csd(npy):
    sizes = np.load(os.path.join(fdir_distr, npy), allow_pickle=False)
    csd, _ = np.histogram(sizes, bin_edge, density=True)
    return csd


def plot_slope(x0, x1, slope, label, dashes, fac, axes):
    for i, ax in enumerate(axes):
        x = np.linspace(x0, x1, 3)
        logx = np.log10(x)
        logy = slope * logx
        y = 10 ** logy
        ax.loglog(x, y * fac, color=plc, linewidth=1, ls='--', label=None)


def plot_sensitivity_panel(fmt='png', dni=True):
    """
    This is the 3-panel plot that visualizes the sensitivity to LES domain and resolution
    :return:
    """
    csd_track_256_50 = make_csd('cs_tracking_256_50.npy')
    csd_track_256_100 = make_csd('cs_tracking_256_100.npy')
    csd_track_512_100 = make_csd('cs_tracking_512_100.npy')

    csd_line_256_50 = make_csd('cs_lines_256_50.npy')
    csd_line_256_100 = make_csd('cs_lines_256_100.npy')
    csd_line_512_100 = make_csd('cs_lines_512_100.npy')

    # load data
    if dni:
        csd_point_bsrn = make_csd('cs_point_bsrn_dni.npy')
        csd_point_256_50 = make_csd('cs_point_256_50_dni.npy')
        csd_point_256_100 = make_csd('cs_point_256_100_dni.npy')
        csd_point_512_100 = make_csd('cs_point_512_100_dni.npy')
    else:
        csd_point_bsrn = make_csd('cs_point_bsrn.npy')
        csd_point_256_50 = make_csd('cs_point_256_50.npy')
        csd_point_256_100 = make_csd('cs_point_256_100.npy')
        csd_point_512_100 = make_csd('cs_point_512_100.npy')

    fig, axes = plt.subplots(1, 3, figsize=gutils.get_image_size(ratio=0.4))

    axes[0].text(0.99, 1.01, 'Point', transform=axes[0].transAxes, ha='right', va='bottom')
    axes[0].plot(bins, f(csd_point_bsrn), color='k', label='BSRN', zorder=5)
    axes[0].plot(bins, f(csd_point_256_50), color='tab:red', )
    axes[0].plot(bins, f(csd_point_256_100), color='tab:red', ls='--')
    axes[0].plot(bins, f(csd_point_512_100), color='tab:red', ls=':')

    plot_slope(600, 10000, -2.7, '-2.7', [4, 4], 6000, axes=axes)
    plot_slope(600, 7000, -1.66, '-1.66', [4, 4], 50, axes=axes)

    axes[2].text(0.99, 1.01, 'Tracking', transform=axes[2].transAxes, ha='right', va='bottom')
    axes[2].plot(bins, f(csd_track_256_50), color='tab:red', label='12.8 km @ 50 m')
    axes[2].plot(bins, f(csd_track_256_100), color='tab:red', dashes=[5, 2], label='25.6 km @ 100 m')
    axes[2].plot(bins, f(csd_track_512_100), color='tab:red', dashes=[2, 2], label='51.2 km @ 100 m')

    axes[1].text(0.99, 1.01, 'Random line', transform=axes[1].transAxes, ha='right', va='bottom')
    axes[1].plot(bins, f(csd_line_256_50), color='tab:red')
    axes[1].plot(bins, f(csd_line_256_100), color='tab:red', dashes=[5, 2])
    axes[1].plot(bins, f(csd_line_512_100), color='tab:red', dashes=[2, 2])

    # plot labels
    fig.legend(bbox_to_anchor=(0.5, 1.02), loc='center', ncol=4, frameon=False, handletextpad=0.5, columnspacing=2,
               handlelength=1.5)

    for ax, label in zip(axes, 'abc'):
        ax.set_yscale('log')
        ax.text(0., 1.01, '$\\bf{%s}$)' % label, transform=ax.transAxes, ha='left', va='bottom')
        ax.set_xlabel('Cloud size (m)')
        ax.set_ylabel(r'Frequency (m$^{-1})$') if label == 'a' else None
        ax.set_ylim(1e-8, 1e-2)
        ax.set_xlim(2e2, 1e5)
        ax.text(2e3, 3e-4, '-1.66', color='gray', ha='left', va='bottom', fontsize=8)
        ax.text(2e3, 3e-6, '-2.70', color='gray', ha='right', va='top', fontsize=8)

    # export and close
    plt.tight_layout()
    plt.savefig(os.path.join(settings.fdir_img_paper2, 'csd_sensitivity.pdf'), bbox_inches='tight', dpi=gsettings.dpi)
    plt.close()


def plot_main_panel(fmt='png', dni=True):
    """
    This figure is a two-panel visualisation of measurement technique based on LES and the circle model
    :return:
    """
    # create figure with 2 axis
    fig, axes = plt.subplots(1, 2, figsize=gutils.get_image_size(ratio=0.6))

    # load data for panel a
    csd_track_512_100 = make_csd('cs_tracking_512_100.npy')
    csd_line_512_100 = make_csd('cs_lines_512_100.npy')
    if dni:
        csd_point_bsrn = make_csd('cs_point_bsrn_dni.npy')
        csd_point_512_100 = make_csd('cs_point_512_100_dni.npy')
    else:
        csd_point_bsrn = make_csd('cs_point_bsrn.npy')
        csd_point_512_100 = make_csd('cs_point_512_100.npy')

    # plot CSD from LES and obs
    axes[0].plot(bins, f(csd_point_bsrn), label='Shadow obs.', color='black', zorder=1)
    axes[0].plot(bins, f(csd_track_512_100), label='Tracking', color='tab:blue', ls='-', zorder=0)
    axes[0].plot(bins, f(csd_point_512_100), label='Point', color='tab:red', ls='-', zorder=0)
    axes[0].plot(bins, f(csd_line_512_100), label='Random line', color='tab:red', ls='--', zorder=0)

    axes[0].legend(ncol=2, bbox_to_anchor=(0.5, 1.03), loc='lower center', frameon=False, handletextpad=0.5,
                   columnspacing=2, handlelength=1.5)

    # add slopes to subplot
    plot_slope(600, 10000, -2.7, '-2.7', [4, 4], 6000, axes=[axes[0]])
    plot_slope(600, 7000, -1.66, '-1.66', [4, 4], 70, axes=[axes[0]])
    axes[0].set_xlabel('Cloud size (m)')

    # load data for panel b
    rc_input = np.load(os.path.join(fdir_distr, 'rc_input.npy'))
    rc_line = np.load(os.path.join(fdir_distr, 'rc_exact.npy'))
    rc_binc = np.load(os.path.join(fdir_distr, 'rc_binc.npy'))

    # plot CSDs
    axes[1].plot(rc_binc, rc_input, label='Input circles', color='tab:blue')
    axes[1].plot(rc_binc, rc_line, label='Random line', color='tab:red')

    axes[1].legend(ncol=2, bbox_to_anchor=(0.5, 1.03), loc='lower center', frameon=False, handletextpad=0.5,
                   columnspacing=2, handlelength=1.5)

    plot_slope(800, 1.5e4, -2.7, '-2.7', [4, 4], 5000, axes=[axes[1]])
    plot_slope(800, 1.5e4, -1.66, '-1.66', [4, 4], 80, axes=[axes[1]])
    axes[1].set_xlabel('Circle size (m)')

    # plot layout
    for ax, label in zip(axes, 'ab'):
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.text(0., 1.01, '$\\bf{%s}$)' % label, transform=ax.transAxes, ha='left', va='bottom')
        ax.set_ylabel(r'Frequency (m$^{-1})$') if label == 'a' else None
        ax.set_ylim(1e-8, 1e-2)
        ax.set_xlim(2e2, 1e5)
        ax.text(2e3, 3e-4, '-1.66', color=plc, ha='left', va='bottom', fontsize=8)
        ax.text(2e3, 3e-6, '-2.70', color=plc, ha='right', va='top', fontsize=8)

    # export and close
    plt.tight_layout()
    plt.savefig(os.path.join(settings.fdir_img_paper2, 'csd_main.%s' % fmt), bbox_inches='tight', dpi=gsettings.dpi)
    plt.close()


if __name__ == "__main__":
    fmt_ = 'pdf'

    plot_sensitivity_panel(fmt=fmt_)
    # plot_main_panel(fmt=fmt_)
