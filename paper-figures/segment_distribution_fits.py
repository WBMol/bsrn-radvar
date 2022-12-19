import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib as mpl
from matplotlib import rc
import os
import numpy as np
import powerlaw

from general import settings as gsettings
from general import utils as gutils
import settings
from segment_distributions import prepare_distribution_data

fontsize = 8
mpl.use('Agg')
rc('font', size=fontsize)

plt.style.use(gsettings.fpath_mplstyle)

fdir_cache = './cache/'
os.makedirs(fdir_cache) if not os.path.exists(fdir_cache) else None


def power_law_sense_seasonal(variable='shadow', cth_threshold=None, fmt='png', sample_interval=1, guess_xmin=False,
                             test_distribution_significance=False):
    """
    Fitting power law to distribution data using the python package. This tests on all data, as
    well as summer and winter subsets. This is the main fitting function, additional sensitivity is
    done separately.

    :param fmt: png
    :param str variable: which variable to load (shadow  or cloud enh)
    :param cth_threshold: the km threshold for cloud top height
    :param int sample_interval: sample interval to take (1 = all, 10 = every 10th sample)
    :param bool guess_xmin: whether to guess xmins (much slower, required once)
    :param bool test_distribution_significance: do a statistical test to compare plexp vs pl
    :return:
    """
    # plotting and fitting parameters
    colors = ['black', 'tab:blue', 'tab:red']
    seasons = ['all', 'winter', 'summer']
    xmaxs = [296e3, 246e3, 341e3]
    xmins = [765, 677, 830]
    guess_xmins = [(i * 0.95, i * 1.05) for i in xmins]

    # create a figure in advance
    fig, axes = plt.subplots(1, 3, figsize=gutils.get_image_size(ratio=0.5), sharey=True)

    for i in range(len(seasons)):
        # load data
        segments = prepare_distribution_data(variable, cth_threshold, seasons[i])

        # prepare data by filtering a subset of segments
        ss = segments.where((segments.dir_min <= 120), drop=False)
        ss = ss.where((segments.duration >= 5), drop=False)
        ss = ss.dropna(dim='segment')
        ss['size'] = ss['duration'] * ss['u200']
        ss = ss.where(ss.size > 1)
        ss = ss.dropna(dim='segment')

        # fit data
        if guess_xmin:
            fit = powerlaw.Fit(ss['size'][::1], xmin=guess_xmins[i], xmax=xmaxs[i])
        else:
            fit = powerlaw.Fit(ss['size'][::sample_interval], xmin=xmins[i], xmax=xmaxs[i])
        print(seasons[i], fit.xmin, fit.data.shape, fit.truncated_power_law.alpha, fit.sigma)

        if test_distribution_significance:
            print(fit.distribution_compare('power_law', 'truncated_power_law'))

        # visualize fit
        fit.plot_pdf(color=colors[i], zorder=5, label=seasons[i], linewidth=2, alpha=0.4, ax=axes[i])
        fit.truncated_power_law.plot_pdf(ax=axes[i], ls='--', color=colors[i], label='fit')
        axes[i].text(.98, .98, '$\\alpha$ = %.3f\n$\\lambda^{-1}$ = %.1f km' %
                     (fit.truncated_power_law.alpha, 1e-3 / fit.truncated_power_law.parameter2),
                     transform=axes[i].transAxes, ha='right', va='top')

    # plot layout
    for ax in axes:
        ax.legend(ncol=2, bbox_to_anchor=(0.5, 1.03), loc='center', frameon=False, handlelength=1.5,
                  handletextpad=0.5, columnspacing=1)
        ax.set_xlabel('Shadow size (m)')
        ax.set_xlim(min(xmins), max(xmaxs))
        ax.set_ylim(1e-9, 1e-3)
    axes[0].set_ylabel('PDF (m$^{-1}$)')

    # export and close
    plt.tight_layout()
    plt.savefig(os.path.join(settings.fdir_img_paper1, 'power_law_fits_seasonal.%s' % fmt), bbox_inches='tight',
                dpi=120)
    plt.close()


def power_law_sense_y2y(variable='shadow', cth_threshold=None, fmt='png', sample_interval=1, guess_xmin=False,
                        test_distribution_significance=False):
    """
    Testing y2y sensitivity of power law fits

    :param fmt: png
    :param str variable: which variable to load (shadow  or cloud enh)
    :param cth_threshold: the km threshold for cloud top height
    :param int sample_interval: sample interval to take (1 = all, 10 = every 10th sample)
    :param bool guess_xmin: whether to guess xmins (much slower, required once)
    :param bool test_distribution_significance: do a statistical test to compare plexp vs pl
    :return:
    """
    # plotting and fitting parameters
    subsets = range(2011, 2021)
    xmaxs = [296e3] * len(subsets)
    xmins = [765] * len(subsets)
    colors = [plt.get_cmap('viridis')(i) for i in np.linspace(0, 1, len(subsets))]
    guess_xmins = [(i * 0.95, i * 1.05) for i in xmins]
    alphas = []
    lambdas = []

    # create a figure in advance
    fig, axes = plt.subplots(1, 1, figsize=gutils.get_image_size(), sharey=True)
    axes = [axes]

    # load data
    segments = prepare_distribution_data(variable, cth_threshold, 'all')

    # prepare data by filtering a subset of segments
    ss = segments.where((segments.dir_min <= 120), drop=False)
    ss = ss.dropna(dim='segment')
    ss['size'] = ss['duration'] * ss['u200']
    ss = ss.where(ss.size > 1)
    ss = ss.dropna(dim='segment')

    for i in range(len(subsets)):
        ss_ = ss.where(segments.start_time.dt.year == subsets[i])
        ss_ = ss_.dropna(dim='segment')

        # fit data
        if guess_xmin:
            fit = powerlaw.Fit(ss_['size'][::sample_interval], xmin=guess_xmins[i], xmax=xmaxs[i])
        else:
            fit = powerlaw.Fit(ss_['size'][::sample_interval], xmin=xmins[i], xmax=xmaxs[i])
        print(subsets[i], fit.xmin, fit.data.shape, fit.truncated_power_law.alpha, fit.sigma)

        alphas.append(fit.truncated_power_law.alpha)
        lambdas.append(1e-3 / fit.truncated_power_law.parameter2)

        if test_distribution_significance:
            print(fit.distribution_compare('power_law', 'truncated_power_law'))

        # visualize fit
        # fit.plot_pdf(color=colors[i], zorder=5, label=subsets[i], linewidth=2, alpha=0.4, ax=axes[0])
        fit.truncated_power_law.plot_pdf(ax=axes[0], ls='-', color=colors[i], label='%i ($\\alpha$ = %.2f)' %
                                                                                    (subsets[i],
                                                                                     fit.truncated_power_law.alpha))

    # plot layout
    for ax in axes:
        ax.legend(ncol=1, bbox_to_anchor=(1.15, 0.5), loc='right', frameon=False, handlelength=1.5,
                  handletextpad=0.5, columnspacing=1)
        ax.set_xlabel('Shadow size (m)')
        ax.set_xlim(min(xmins), max(xmaxs))
        ax.set_ylim(1e-9, 1e-3)
    axes[0].set_ylabel('PDF (m$^{-1}$)')

    # export and close
    plt.tight_layout()
    plt.savefig(os.path.join(settings.fdir_img_paper1, 'power_law_fits_y2y.%s' % fmt), bbox_inches='tight',
                dpi=120)
    plt.close()

    # print alpha results
    print('%.3f +/- %.3f' % (np.mean(alphas), np.std(alphas)))
    print('%.3f +/- %.3f' % (np.mean(lambdas), np.std(lambdas)))


def power_law_sense_sea(variable='shadow', cth_threshold=None, fmt='png', sample_interval=1, guess_xmin=False,
                        test_distribution_significance=False):
    """
    Test solar elevation angle sensitivity to fits

    :param fmt: png
    :param str variable: which variable to load (shadow  or cloud enh)
    :param cth_threshold: the km threshold for cloud top height
    :param int sample_interval: sample interval to take (1 = all, 10 = every 10th sample)
    :param bool guess_xmin: whether to guess xmins (much slower, required once)
    :param bool test_distribution_significance: do a statistical test to compare plexp vs pl
    :return:
    """
    # plotting and fitting parameters
    colors = ['tab:blue', 'tab:red']
    subsets = ['low', 'high']
    xmaxs = [1e5] * 2
    xmins = [761] * 2
    guess_xmins = [(i * 0.95, i * 1.05) for i in xmins]

    # create a figure in advance
    fig, axes = plt.subplots(1, 2, figsize=gutils.get_image_size(ratio=0.5), sharey=True)

    # load data
    segments = prepare_distribution_data(variable, cth_threshold, 'summer')

    # prepare data by filtering a subset of segments
    ss = segments.where((segments.dir_min <= 120), drop=False)
    ss = ss.dropna(dim='segment')
    ss['size'] = ss['duration'] * ss['u200']
    ss = ss.where(ss.size > 1)

    for i in range(len(subsets)):
        if subsets[i] == 'low':
            ss_ = ss.where((segments.solar_angle < 40), drop=False)
        elif subsets[i] == 'high':
            ss_ = ss.where((segments.solar_angle > 45), drop=False)
        else:
            raise NotImplementedError('subsets other than high or low not implemented.')
        ss_ = ss_.dropna(dim='segment')

        # fit data
        if guess_xmin:
            fit = powerlaw.Fit(ss_['size'][::sample_interval], xmin=guess_xmins[i], xmax=xmaxs[i])
        else:
            fit = powerlaw.Fit(ss_['size'][::sample_interval], xmin=xmins[i], xmax=xmaxs[i])
        print(subsets[i], fit.xmin, fit.data.shape, fit.truncated_power_law.alpha, fit.sigma)

        if test_distribution_significance:
            print(fit.distribution_compare('power_law', 'truncated_power_law'))

        # visualize fit
        fit.plot_pdf(color=colors[i], zorder=5, label=subsets[i], linewidth=2, alpha=0.4, ax=axes[i])
        fit.truncated_power_law.plot_pdf(ax=axes[i], ls='--', color=colors[i], label='fit')
        axes[i].text(.98, .98, '$\\alpha$ = %.3f\n$\\lambda^{-1}$ = %.1f km' %
                     (fit.truncated_power_law.alpha, 1e-3 / fit.truncated_power_law.parameter2),
                     transform=axes[i].transAxes, ha='right', va='top')

    # plot layout
    for ax in axes:
        ax.legend(ncol=2, bbox_to_anchor=(0.5, 1.03), loc='center', frameon=False, handlelength=1.5,
                  handletextpad=0.5, columnspacing=1)
        ax.set_xlabel('Shadow size (m)')
        ax.set_xlim(min(xmins), max(xmaxs))
        ax.set_ylim(1e-9, 1e-3)
    axes[0].set_ylabel('PDF (m$^{-1}$)')

    # export and close
    plt.tight_layout()
    plt.savefig(os.path.join(settings.fdir_img_paper1, 'power_law_fits_sea.%s' % fmt), bbox_inches='tight',
                dpi=120)
    plt.close()


def calculate_shadow_size(cloud_size, transparent_edge_size=45):
    """
    Convert cloud size to shadow size based on a model that includes some kind of size of the cloud edge size

    :param np.ndarray cloud_size: size in meters of the cloud
    :param transparent_edge_size: length scale to subtract for 1 cloud edge (meters)
    :return:
    """
    # retrieve cloud duration first, to subtract a bit of time, and convert back
    cloud_size = cloud_size - 5 * 10

    # calculate the new cloud sizes
    shadow_size = cloud_size - (2 * transparent_edge_size)
    shadow_size[shadow_size < 0] = 0
    return shadow_size


def simulate_cloud_and_shadows(xmin=10, n=1e5, cache=True, add_uncertainty=False, lce=50):
    """
    Simulate a shadow size distribution based on a cloud size distribution

    :param xmin: minimal size of the analysed distribution
    :param n: number of random clouds to sample. Only applied when generating a new distribution.
    :param bool cache: whether to take the randomly generated clouds from cache or not.
    :param bool add_uncertainty: whether to add a x2 and /2 value for cloud edge size
    :param int lce: length scale for cloud edge transparency
    :return:
    """
    # create a power law based on the cloud size distribution
    csd_wf = powerlaw.Power_Law(xmin=xmin, xmax=1e5, parameters=[1.66])
    csd_mol = powerlaw.Truncated_Power_Law(xmin=xmin, xmax=1e5, parameters=[1.647, 1 / 81.2e3])
    csd_mol_gen = powerlaw.Power_Law(xmin=xmin, xmax=1e5, parameters=[1.643])
    csd_xs = np.logspace(np.log10(xmin), 5, 31)
    csd_ys = csd_wf.pdf(data=csd_xs)
    csd2_ys = csd_mol.pdf(data=csd_xs)

    # get generated clouds from a CSD
    fpath_cache = os.path.join('./cache/', 'plt_1e5.npy')
    if cache and os.path.isfile(fpath_cache):
        print("Loading generated CSD from cache")
        csd_data = np.load(fpath_cache)
    else:
        print("Generating CSD from power law")
        csd_data = csd_mol_gen.generate_random(n=int(n))
        if cache:
            print("Exporting CSD to cache")
            np.save(fpath_cache, csd_data)
    print("Randomly generated clouds from CSD are ready.")

    # fit those clouds back to use the class instance for later
    # csd_fit = powerlaw.Fit(csd_data, xmin=xmin, xmax=1e5)

    # visualize the simulated distribution
    fig = plt.figure(figsize=gutils.get_image_size())
    axes = [fig.add_subplot(1, 2, 1)]
    # fig, axes = plt.subplots(1, 2, figsize=gutils.get_image_size(ratio=0.5))
    ax = axes[0]

    # ax.plot(csd_xs, csd_ys * 8, color='gray', label='Clouds ($\\alpha = 1.66$)', ls='--', zorder=1)
    ax.plot(csd_xs[7:], csd_ys[7:] * 14, color='gray', ls='--', zorder=1, lw=1.)
    ax.text(2e3, 1e-3, '$x^{-1.66}$ clouds', color='gray', ha='left', va='bottom')
    # ax.text(1e3, 1e-3, 'Wood & Field (2010)', color='gray', ha='left', va='bottom', fontsize=9)
    ax.plot(csd_xs[14:], csd2_ys[14:] * 4.5, color='black', label='Best fit', alpha=0.8, lw=1., ls='-', zorder=5)
    ax.plot(csd_xs[:15], csd2_ys[:15] * 4.5, color='black', ls='--', alpha=0.8, lw=1., zorder=4)

    # convert cloud sizes to shadow sizes
    ssd_data = calculate_shadow_size(csd_data, transparent_edge_size=lce)
    ssd_fit = powerlaw.Fit(ssd_data, xmin=xmin, xmax=1e5)
    ssdx, ssdy = ssd_fit.pdf()
    ssdx = (ssdx[1:] + ssdx[:-1]) / 2
    ax.plot(ssdx, ssdy, label='Model', color='#EB0101', lw=1., zorder=4, ls='-')

    if add_uncertainty:
        for lce_ in [lce / 2, lce * 2]:
            ssd_data = calculate_shadow_size(csd_data, transparent_edge_size=lce_)
            ssd_fit = powerlaw.Fit(ssd_data, xmin=xmin, xmax=1e5)
            ssdx_, ssdy_ = ssd_fit.pdf()
            ssdx_ = (ssdx_[1:] + ssdx_[:-1]) / 2
            ax.plot(ssdx_, ssdy_ * (ssdy[-1] / ssdy_[-1]), color='tab:orange', lw=1., zorder=2, alpha=0.8, ls='-')

    # load observed ssd
    segments = prepare_distribution_data(variable='shadow', season='all', cth_threshold=None)
    ss = segments.where((segments.dir_min <= 120), drop=False)
    ss = ss.dropna(dim='segment')
    ss['size'] = (ss['duration']) * ss['u200']
    ss = ss.where(ss.size > 1, drop=False)
    # ss = ss.where(ss['u200'], drop=False)
    ss = ss.dropna(dim='segment')
    obs_fit = powerlaw.Fit(ss['size'][::1], xmin=761, xmax=342e3)
    obsx, obsy = obs_fit.pdf(original_data=True)
    obsx = (obsx[1:] + obsx[:-1]) / 2.
    cobs = plt.get_cmap('viridis')(0) if False else '#1799EB'
    ax.plot(obsx, obsy, label='Obs', color=cobs, lw=3, zorder=1, alpha=1.)

    # plot layout
    ax.set_xlabel('Shadow size (m)')
    ax.set_ylabel('Probability density (m$^{-1}$)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    # fig.legend(bbox_to_anchor=(0.5, 1.01), loc='lower center', ncol=2, frameon=False)
    ax.legend(bbox_to_anchor=(0.5, 1.03), loc='lower center', ncol=3, frameon=False, columnspacing=1, handlelength=1.5,
              handletextpad=.5)
    ax.set_ylim(1e-8, 1e-1)
    ax.set_xlim(xmin, 1e5)
    ax.set_yticks(np.logspace(-8, -1, 8))
    ax.set_xticks(np.logspace(1, 5, 5))

    arrow = mpatches.FancyArrowPatch((0.125, 0.81), (0.125, 0.95), color='black',
                                     mutation_scale=10, arrowstyle='<->',
                                     transform=ax.transAxes)
    ax.add_patch(arrow)
    arrow = mpatches.FancyArrowPatch((0.66, 0.71), (0.57, 0.62), color='gray',
                                     mutation_scale=10, arrowstyle='-|>', linestyle='-',
                                     transform=ax.transAxes)
    ax.add_patch(arrow)

    # add image
    pic = plt.imread('./cloud_edge.jpg')
    axes.append(fig.add_subplot(2, 2, 2))
    axes[1].imshow(pic)
    axes[1].set_ylim(900, 275)
    # axes[1].set_xlim(100, 100 + 1065)
    axes[1].axis('off')
    # axes[1].set_xticks([])
    # axes[1].set_yticks([])

    # add diagram
    axes.append(fig.add_subplot(2, 2, 4))

    from scripts.timeseries.analyse_bsrn_response_time import simulate_idealised_response
    x, yp, yc = simulate_idealised_response(return_values=True, c_dur=20, c_start=10)
    axes[2].plot(x, yc, color='#EB0101', lw=1., ls='-', label='Instant')
    axes[2].plot(x, yp, color='tab:blue', lw=1., ls='-', label='Pyrheliometer')
    axes[2].plot([10.5, 29.5], [800, 800], color='gray', lw=5)
    axes[2].text(20., 770, 'cloud', color='gray', ha='center', va='top')
    axes[2].plot([10, 10], [0, 800], color='gray', ls=':', lw=1, zorder=0)
    axes[2].plot([30, 30], [0, 800], color='gray', ls=':', lw=1, zorder=0)
    # axes[2].plot([0, 50], [120, 120], ls='--', color='gray')
    axes[2].set_ylim(0, 810)
    axes[2].set_yticks([0, 250, 500, 750])
    axes[2].set_xticks(range(0, 51, 10))
    axes[2].set_xlim(0, 50)
    # axes[2].text(49, 120, 'shadow\nthreshold', ha='right', va='center', color='gray')
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('Direct irradiance (W m$^{-2}$)')
    axes[2].legend(ncol=3, bbox_to_anchor=(0.5, 1.03), loc='lower center', frameon=False, columnspacing=1,
                   handlelength=1.5, handletextpad=.5)

    for ax, label in zip(axes, 'abc'):
        ax.text(0., 1.01 if label == 'a' else 1.02, '$\\bf{%s}$)' % label, transform=ax.transAxes, ha='left', va='bottom')

    # export and close
    plt.tight_layout()
    plt.savefig(os.path.join(settings.fdir_img_paper2, 'distribution_fit_and_simulation.pdf'), bbox_inches='tight',
                dpi=300)
    plt.close()


if __name__ == "__main__":
    power_law_sense_seasonal(fmt='png', sample_interval=1, guess_xmin=False)
    power_law_sense_y2y(fmt='png', sample_interval=1)
    power_law_sense_sea(fmt='png', sample_interval=1)
    simulate_cloud_and_shadows(xmin=10, n=1e7, cache=False, add_uncertainty=False, lce=25)
