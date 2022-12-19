import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os
import shapely.geometry as sgeom
from cartopy.geodesic import Geodesic
from matplotlib import rc

import settings
from general import utils as gutils
from general import settings as gsettings

plt.style.use(gsettings.fpath_mplstyle)
rc('font', size=settings.fontsize)

locs = {
    'bsrn': (4.927740, 51.968063),
    'tower': (4.926343144805244, 51.970312667487704)
}

# define projs
proj = ccrs.Orthographic(central_latitude=locs['bsrn'][1], central_longitude=locs['bsrn'][0])
xproj = ccrs.PlateCarree()


def plot_bsrn_dataset_location(fmt='png', add_img=False):
    """

    :param fmt: image output format (png or pdf)
    :param bool add_img: whether to add a cabauw image as subplot to it
    :return:
    """
    # create fig, axes
    fig = plt.figure(figsize=gutils.get_image_size(text_width=1 if add_img else 0.5), constrained_layout=True)
    ax_map = fig.add_subplot(121 if add_img else 111, projection=proj)
    ax_img = fig.add_subplot(122) if add_img else None
    axes = [ax_map, ax_img]

    # add data
    ax_map.scatter(locs['bsrn'][0], locs['bsrn'][1], transform=xproj, c='tab:blue', marker='x', s=15)

    # plot layout
    dlonb, dlons = 3.6, 0.8
    dratio = 2.2
    ax_map.set_extent([locs['bsrn'][0] - dlonb, locs['bsrn'][0] + dlonb,
                        locs['bsrn'][1] - dlonb / dratio, locs['bsrn'][1] + dlonb / dratio], crs=xproj)

    gl = ax_map.gridlines(draw_labels=True, zorder=0, alpha=0.5)
    gl.top_labels = False
    gl.right_labels = False
    ax_map.coastlines()

    # ax_map.add_feature(cfeature.LAND)
    # ax_map.add_feature(cfeature.OCEAN)
    ax_map.add_feature(cfeature.BORDERS, linestyle='-', alpha=0.7)
    # ax_map.add_feature(cfeature.LAKES, alpha=0.8)
    # ax.add_feature(cfeature.RIVERS)

    gd = Geodesic()
    geoms = []
    for r in [5e3, 10e3, 15e3]:
        cp = gd.circle(lon=locs['bsrn'][0], lat=locs['bsrn'][1], radius=r)
        geoms.append(sgeom.Polygon(cp))
    axes[0].add_geometries(geoms, crs=xproj, edgecolor='tab:red', alpha=0.6, facecolor='none')

    # add image
    if add_img:
        img = plt.imread('./tower_bsrn.jpg')
        ax_img.imshow(img)
        ax_img.axis('off')

        for ax, label in zip(axes, 'ab'):
            ax.text(0., 1.01, '$\\bf{%s}$)' % label, transform=ax.transAxes, ha='left', va='bottom')

    # export and close
    fpath_out = os.path.join(settings.fdir_img_paper1, 'bsrn_location.%s' % fmt)
    plt.savefig(fpath_out, bbox_inches='tight', dpi=300)
    plt.close()


if __name__ == "__main__":
    plot_bsrn_dataset_location(fmt='pdf', add_img=True)
