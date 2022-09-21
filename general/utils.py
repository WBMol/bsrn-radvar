from datetime import timedelta
import xarray
import os

from general import settings
from scripts.processor import settings as psettings


def generate_dt_range(dt_start, dt_stop, delta_dt=None, inclusive=False):
    """
    Generate a list of datetimes ranging from start to stop with delta_dt interval

    :param dt_start: start datetime (inclusive)
    :param dt_stop:  stop datetime (exclusive by default)
    :param delta_dt:  timedelta timestep to take
    :param inclusive: whether to include the dt_stop in the result
    :return:
    """
    dts = []
    delta_dt = timedelta(days=1) if delta_dt is None else delta_dt
    while dt_start < dt_stop:
        dts.append(dt_start)
        dt_start += delta_dt

    if inclusive:
        dts.append(dt_stop)

    return dts


def get_image_size(page_width=6.69, ratio=0.61, text_width=1.):
    """
    return image size based on some basic latex parameters

    :param page_width: width of latex document page in inches
    :param text_width: text width fraction of full page with in latex document
    :param ratio: height / width ratio (default is golden ratio
    :return:
    """
    # figure latex settings (for publication)
    img_h = page_width * text_width
    img_v = img_h * ratio
    return img_h, img_v


def load_timeseries_data(fpath, apply_quality_control=True):
    """

    :param fpath: full filepath to processed irradiance data
    :param apply_quality_control: if 1sec data, then by default apply to custom quality filter to basic vars
    :return:
    """
    # load dataset
    data = xarray.open_dataset(fpath)

    if apply_quality_control:
        if data.attrs['resolution'] == '1sec':
            # apply quality control
            for var in ['ghi', 'dif', 'dhi_sac', 'dhi_sac', 'dni']:
                data[var] = data[var].where(data['custom_sw_quality'] == 0)
            # data['lwd'] = data['lwd'].where(data['custom_lw_quality'] == 0)
    return data


def generate_processed_fpath(date, res='1min', which='bsrn', filter_name=None, raw=False):
    """
    Function to generate filefpaths of processed data. Files that are raw or otherwise unprocessed need to be
    procesed first. This function is purely for processed data.

    :param date: datetime to retrieve data for
    :param str res: which resolution dataset, 1min or 1sec, on applies to bsrn
    :param str which: main paths of the ones for debugging
    :param str filter_name: optional for debugging classification data
    :return:
    """
    if which == 'bsrn':
        fdir = psettings.fdir_out.format(res=res, y=date.year, m=date.month)
        fname = date.strftime('%Y%m%d.nc')
        return os.path.join(fdir, fname)
    elif which == 'debug':
        if filter_name is not None:
            return os.path.join(res, date.strftime('%Y%m%d') + '_%s.nc' % filter_name)
        else:
            raise Exception("Should provide a valid filter name when requesting debug filepaths")
    elif which == 'msgcpp':
        fdir = os.path.join(settings.fdir_research_data, 'eumetsat', 'msgcpp', 'processed', 'timeseries',
                            str(date.year), date.strftime('%m'))
        fname = date.strftime('%Y%m%d.nc')
        return os.path.join(fdir, fname)
    elif which == 'nubiscope':
        if raw:
            fdir = os.path.join(settings.fdir_research_data, 'Cabauw', 'Nubiscope', 'raw', str(date.year))
            fname = date.strftime('cesar_nubiscope_cloudcover_la1_t10_v1.0_%Y%m.nc')
        else:
            fdir = os.path.join(settings.fdir_research_data, 'Cabauw', 'Nubiscope', 'processed', str(date.year),
                                date.strftime('%m'))
            fname = date.strftime('%Y%m%d.nc')
        return os.path.join(fdir, fname)
    elif which == 'class_validation':
        fdir = os.path.join(settings.fdir_research_data, 'Cabauw', 'BSRN', 'class_validation', str(date.year),
                            date.strftime('%m'))
        fname = date.strftime('%Y%m%d.nc')
        return os.path.join(fdir, fname)
    else:
        raise NotImplementedError('Type %s is unsupported' % which)
