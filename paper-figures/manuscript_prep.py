import os
import settings
from shutil import copyfile

figure_order = {
    'bsrn_essd': [
        'bsrn_location',
        'bsrn_1sec_availability',
        'cloud_fraction_compare',
        '20150418',
        '20150403',
        '20160815',
        'classification_climate',
        'irradiance_climate_1sec',
        'class_climatology_overview',
    ]
}


def rename_figures(manuscript):
    if manuscript == 'bsrn_essd':
        fdir = settings.fdir_img_paper1
    else:
        raise NotImplementedError('No settings for manuscript %s' % manuscript)

    for i, figure_name in enumerate(figure_order[manuscript]):
        fpath_in = os.path.join(fdir, figure_name + '.pdf')
        fpath_out = os.path.join(fdir, 'fig{i:02}.pdf'.format(i=i+1))
        copyfile(fpath_in, fpath_out)


if __name__ == "__main__":
    rename_figures(manuscript='bsrn_essd')
