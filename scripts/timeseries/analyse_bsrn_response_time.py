import matplotlib.pyplot as plt
import os
import numpy as np

from general import settings as gsettings
from general import utils as gutils
import settings

plt.style.use(gsettings.fpath_mplstyle)


def simulate_idealised_response(fmt='png', return_values=False, q_dir=750, dur=50, c_dur=25, c_start=5):
    """
    Create a time series that simulates a sensor's exponential response with certain time lag
    :return:
    """
    # some settings
    dt = 0.1
    n_sec = int(dur / dt)
    x_ax = np.arange(0, dur, dt)
    c_dur = int(c_dur / dt)
    d_ce = int(5 / dt)
    ci = int(c_start / dt)
    cj = ci + c_dur

    # create block function
    qdir_block = np.ones(n_sec) * q_dir
    qdir_block[ci:cj] = 0

    qdir_real_tau = 1 - qdir_block.copy() / q_dir
    qdir_real_tau[ci:ci + d_ce] = np.linspace(0, 1, d_ce)
    qdir_real_tau[cj - d_ce:cj] = np.linspace(0, 1, d_ce)[::-1]
    qdir_real_tau *= 6

    qdir_real = q_dir / np.exp(qdir_real_tau)

    # creates simulated response with exponential response
    qdir_bres = []
    qdir_rres = []
    qdir_7s = []
    qdir_10s = []
    for i in range(n_sec):
        if i == 0:
            qdir_bres.append(qdir_block[i])
            qdir_rres.append(qdir_real[i])
        else:
            delta_q = qdir_block[i] - qdir_bres[-1]
            qdir_bres.append(qdir_bres[-1] + dt * delta_q / np.exp(0.88))

            delta_q = qdir_real[i] - qdir_rres[-1]
            qdir_rres.append(qdir_rres[-1] + dt * delta_q / np.exp(0.88))

        if i < int(7 / dt):
            qdir_7s.append(qdir_block[i])
        else:
            delta_q = qdir_block[i] - qdir_7s[i - int(7 / dt)]
            qdir_7s.append(qdir_7s[i - int(7 / dt)] + delta_q * 0.95)

        if i < int(10 / dt):
            qdir_10s.append(qdir_block[i])
        else:
            delta_q = qdir_block[i] - qdir_10s[i - int(10 / dt)]
            qdir_10s.append(qdir_10s[i - int(10 / dt)] + delta_q * 0.99)

    if return_values:
        return (x_ax, qdir_rres, qdir_real)
    else:
        # create figure and axis
        fig, ax = plt.subplots(1, 1, figsize=gutils.get_image_size(ratio=0.6))

        # visualize results
        # ax.plot(x_ax, qdir_block, color='black', lw=1.5, alpha=0.7, label='block cloud', zorder=1)
        # ax.plot(x_ax, qdir_bres, color='black', lw=1.5, alpha=0.7, ls='--', label='sim. pyrhelio', zorder=2)

        ax.plot(x_ax, qdir_real, color='tab:blue', lw=1.5, label='smooth cloud')
        ax.plot(x_ax, qdir_rres, color='tab:blue', lw=1.5, ls='--', label='sim. pyrhelio', zorder=2)

        # ax.plot(x_ax, qdir_7s, color='black', lw=1, label='95% crit', zorder=0)
        # ax.plot(x_ax, qdir_10s, color='black', label='99% crit', zorder=0)

        # plot settings
        ax.set_ylabel('Direct irradiance (W m$^{-2}$)')
        ax.set_xlabel('Time (seconds)')
        ax.plot([0, n_sec], [120, 120], ls=':', color='black')
        ax.legend(bbox_to_anchor=(0.5, 1.05), loc='lower center', frameon=False, ncol=4)
        ax.set_ylim(0, q_dir * 1.01)
        ax.set_xlim(0, dur)

        print(np.sum(np.array(qdir_rres) < 120) * dt)

        # export and close
        plt.tight_layout()
        plt.savefig(os.path.join(settings.fdir_images, 'simulated_sensor_response.%s' % fmt), bbox_inches='tight',
                    dpi=gsettings.dpi)
        plt.close()


if __name__ == "__main__":
    simulate_idealised_response(fmt='pdf', return_values=True)
