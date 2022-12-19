import xarray
import os
import matplotlib.pyplot as plt
from matplotlib import rc

from general import utils as gutils
from general import settings as gsettings
import settings

plt.style.use(gsettings.fpath_mplstyle)
fdir_stat_file = os.path.join(gsettings.fdir_bsrn_data, '1sec', 'statistics')
rc('font', size=settings.fontsize)

months = range(1, 13)
months_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']


def availability_of_data_overview(data, total=True, absolute=True, fmt='png'):
    """
    Create simple overview plot of data availability for bsrn 1sec data

    :param data: the aggregated data
    :param bool total: if total then just plot a bar for each month, else plot 2D to show year vs. month
    :param bool absolute: whether to plot absolute amount of days or relative w.r.t. maximum
    :param str fmt: output format
    :return:
    """
    # prepare data
    if total:
        data_gr = data.fr_all.groupby('date.month')
    else:
        data_gr = data.to_dataframe()
        data_gr = data_gr.groupby([data_gr.index.year, data_gr.index.month]).mean().rename_axis(('year', 'month'))
        data_gr = data_gr.to_xarray().fr_all

    # create plot
    fig, ax = plt.subplots(1, 1, figsize=gutils.get_image_size(text_width=0.75))

    if total:
        # prepare data for plot
        y = data_gr.sum() if absolute else data_gr.mean() * 100
        x = y.month

        # plot
        ax.bar(x, y, zorder=1)
        ax.set_ylabel('Days' if absolute else '% of max')
        ax.grid(axis='x', zorder=0, alpha=0.3)
    else:
        # prepare axis for plot
        x = data_gr.month
        y = data_gr.year

        # plot
        cmap = plt.get_cmap('RdBu', 20)
        pcm = ax.pcolormesh(x, y, data_gr * 100, vmin=0, vmax=100, cmap=cmap, zorder=0, shading='nearest',
                            rasterized=True)
        ax.set_ylim(2019.5, 2010.5)
        ax.set_yticks(y.values)
        ax.set_yticks(y.values + 0.5, minor=True)
        ax.set_xticks(x.values + 0.5, minor=True)
        ax.grid(which='minor', alpha=0.4, zorder=5)
        fig.colorbar(pcm, ax=ax, shrink=0.75, aspect=20, label='Availability (%)')

        for x_ in x:
            for y_ in y:
                ax.text(x_.values, y_.values, '%.0f' % round((data_gr.sel(month=x_, year=y_).values * 100), 0), color='white', zorder=5,
                        ha='center', va='center')

    # layout
    # ax.set_xlabel('Month of year')
    ax.set_xticks(x, labels=months_labels)

    # export and close
    plt.tight_layout()
    plt.savefig(os.path.join(settings.fdir_img_paper1, 'bsrn_1sec_availability.%s' % fmt),
                bbox_inches='tight', dpi=gsettings.dpi)
    plt.close()


if __name__ == "__main__":
    data = xarray.open_dataset(os.path.join(fdir_stat_file, 'daily_stats_bsrn_1sec.nc'))
    availability_of_data_overview(data, total=False, absolute=False, fmt='pdf')