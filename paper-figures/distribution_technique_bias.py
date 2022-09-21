import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
import os

from general import settings as gsettings
from general import utils as gutils
from scripts.processor import utils as putils
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


def plot_sensitivity_panel(fmt='png'):
    """
    This is the 3-panel plot that visualizes the sensitivity to LES domain and resolution
    :return:
    """

    # load data
    csd_track_256_50 = make_csd('cs_tracking_256_50.npy')
    csd_track_256_100 = make_csd('cs_tracking_256_100.npy')
    csd_track_512_100 = make_csd('cs_tracking_512_100.npy')

    csd_line_256_50 = make_csd('cs_lines_256_50.npy')
    csd_line_256_100 = make_csd('cs_lines_256_100.npy')
    csd_line_512_100 = make_csd('cs_lines_512_100.npy')

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


def plot_main_panel(fmt='png'):
    """
    This figure is a two-panel visualisation of measurement technique based on LES and the circle model
    :return:
    """
    # create figure with 2 axis
    fig, axes = plt.subplots(1, 2, figsize=gutils.get_image_size(ratio=0.6))

    # load data for panel a
    csd_track_512_100 = make_csd('cs_tracking_512_100.npy')
    csd_line_512_100 = make_csd('cs_lines_512_100.npy')
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


@putils.timeit
def random_number_custom(xmin, xmax, n=1000, alpha=-1.7, lda=None):
    from numpy import random

    if lda is None:
        def p_pdf(x):
            return x ** alpha
    else:
        def p_pdf(x):
            return x ** alpha * np.exp(-x/lda)

    # generate random sizes between xmin, xmax
    xs = random.randint(xmin, xmax, size=n)

    # calculate the pdf for each size, and draw random numbers
    xs_prob = p_pdf(xs) / p_pdf(xmin)
    # ps = random.random_sample(size=n)
    ps = random.random_sample(size=n // 4)
    ps = np.concatenate([ps] * 4)

    # generate valid samples
    samples = np.where(xs_prob > ps, xs, np.nan)
    include = np.isfinite(samples)
    print("Requested %i samples, %i found (%.2f%%)" % (n, int(np.sum(include)), int(np.sum(include)) / n * 100))
    return samples[include], p_pdf


def rodts_vs_bart(fmt='png', n=1e5, alpha=1.77, lda=80e3):
    """
    Visualize the difference in 1D vs. 2D between what Rodts et al. say compared to Bart's analysis

    Aka, 1D vs. 2D number densities or contribution to cloud cover
    :return:
    """
    import powerlaw

    # set the x-space over which to visualize results
    xmin, xmax = (1e2, 1e5)
    x = np.logspace(np.log10(xmin), np.log10(xmax), 31)
    x_center = (x[1:] + x[:-1]) / 2

    # generate a power law distribution
    csd = powerlaw.Power_Law(xmin=xmin, xmax=xmax, parameters=[alpha])
    # csd = powerlaw.Truncated_Power_Law(xmin=xmin, xmax=xmax, parameters=[1.7, 1/3.3e3])
    samples = csd.generate_random(n=int(n))

    # generate another custom one
    samples_custom, dfunc = random_number_custom(xmin=xmin, xmax=xmax, n=int(n*5e2), alpha=-alpha, lda=lda)

    # create a 3-panel figure
    fig, axes = plt.subplots(1, 3, figsize=gutils.get_image_size(text_width=1.5, ratio=0.35))

    # visualize size distribution
    pdf_y, _ = np.histogram(samples, bins=x, density=True)
    axes[0].plot(x_center, pdf_y, label='$\\alpha$ = %.2f' % alpha)

    pdf_y_new, _ = np.histogram(samples_custom, bins=x, density=True)
    axes[0].plot(x_center, pdf_y_new, label='$\\alpha$ = %.2f, $\\lambda$ = %.1f km' % (alpha, lda/1e3))

    # visualize diameter to cloud cover contribution based on circle model
    circle_areas = np.pi * (x_center / 2) ** 2
    cc_contrib = circle_areas / circle_areas.sum()
    axes[1].plot(x_center, cc_contrib * 100, color='slategray')

    # cloud cover distribution
    pdf_y_cc = pdf_y * cc_contrib
    pdf_y_cc_new = pdf_y_new * cc_contrib
    pdf_y_an = dfunc(x_center)
    pdf_y_an_cc = pdf_y_an / np.sum(pdf_y_an * np.diff(x)) * cc_contrib

    axes[2].plot(x_center, pdf_y_cc, color='tab:blue', label='$x^{\\alpha}$')
    axes[2].plot(x_center, pdf_y_cc_new, color='tab:orange', label='$x^{\\alpha}$e$^{-x\\lambda}$')
    axes[2].plot(x_center, pdf_y_an_cc, color='tab:orange', ls='--', label='$x^{\\alpha}$e$^{-x\\lambda}$ (al)')

    # plot layout
    axes[0].text(0.5, 1.08, 'Number density 1D', ha='center', va='bottom', transform=axes[0].transAxes)
    axes[1].text(0.5, 1.08, 'Contribution to cloud cover', ha='center', va='bottom', transform=axes[1].transAxes)
    axes[2].text(0.5, 1.08, 'Cloud cover density 1D', ha='center', va='bottom', transform=axes[2].transAxes)

    axes[0].set_ylabel('m$^{-1}$')
    axes[1].set_ylabel('%')
    axes[2].set_ylabel('a.u.')

    axes[0].legend()
    axes[2].legend()

    for ax, label in zip(axes, 'abc'):
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_xlabel('chord length, diameter (m)')
        ax.set_xticks(np.logspace(np.log10(xmin), np.log10(xmax), int(np.log10(xmax/xmin))+1))
        ax.set_xlim(xmin, xmax)
        ax.text(0., 1.01, '$\\bf{%s}$)' % label, transform=ax.transAxes, ha='left', va='bottom')

    # export and close
    plt.tight_layout()
    plt.savefig(os.path.join(settings.fdir_img_paper2, 'rodts_vs_bart_a%.2f.%s' % (alpha, fmt)), dpi=300,
                bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    fmt_ = 'pdf'

    # plot_sensitivity_panel(fmt=fmt_)
    # plot_main_panel(fmt=fmt_)
    rodts_vs_bart(fmt=fmt_, n=1e5, alpha=2.7)
    # random_number_custom(xmin=10, xmax=1000, n=int(5e7))
