import numpy as np
from datetime import datetime, timedelta
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import xarray
import warnings

from general import settings as gsettings
from general import utils as gutils
from scripts.classifier import preprocess_validation

mpl.use('Agg')
plt.style.use(gsettings.fpath_mplstyle)


def find_segments(classes):
    """
    Find the sequences in time series of similar classes

    :param classes: the classification array of a time series
    :return:
    """
    # first, hide 'no-class' and then get the diffs, this signifies points at which a change in class occurs
    classes = xarray.where(classes == 1, np.nan, classes)
    diffs = np.diff(classes)

    # convert the diffs array to indices where changes occur
    changes = np.where(np.abs(diffs) > 0)[0] + 1

    # split the classes by these indices to group together the segments of repeating numbers
    segs = np.split(classes, changes)

    # filter out any segment that contains nans, indicating that the segments got muddied with missing data
    segs_clean = [i for i in segs if i.isnull().sum() == 0]

    # clean up the segments at the beginning and end of the day, they *should* be filtered
    # segs_clean = [segs[0].dropna(dim='time_class')] + segs_clean + [segs[-1].dropna(dim='time_class')]
    return segs_clean


def stats_dict_to_xarray(stats):
    """
    Convert the dictionary with statistics results for event to an xarray dataset with coordinates

    :param stats: event stats in dictionary format
    :return:
    """
    # convert dictionary entries to tuples with coordinate name
    coord = list(range(len(stats['duration'])))
    for stat in stats:
        if stat in ['cth', 'cot', 'cloud_class']:
            stats[stat] = (['segment', 'radius'], stats[stat])
        else:
            stats[stat] = ('segment', stats[stat])

    # create xarray dataset
    dset = xarray.Dataset(data_vars=stats, coords=dict(segment=coord, radius=preprocess_validation.msg_radii))

    # define attributes for the xarray dataset
    long_names = {
        'duration': 'segment length',
        'ghi_max_abs': 'max GHI',
        'ghi_mean_abs': 'mean GHI',
        'ghi_max_rel': 'max GHI difference w.r.t. clear-sky',
        'ghi_mean_rel': 'mean GHI difference w.r.t. clear-sky',
        'ce_sum': 'integrated enhanced GHI w.r.t. clear-sky',
        'ce_max': 'max enhanced GHI w.r.t. clear-sky',
        'dif_wrt_cs': 'mean diffuse fraction w.r.t. clear-sky',
        'dif_wrt_ghi': 'mean diffuse fractin w.r.t. GHI',
        'start_time': 'starting datetime in utc of segment',
        'solar_angle': 'mean solar elevation angle',
        'dir_min': 'minimum direct solar irradiance',
        'class_g1': 'most common weather class (classes_group_1)',
        'u200': 'mean wind speed at 200 meter',
        'cth': 'mean of segment msgcpp area max cloud top height',
        'cot': 'mean of segment msgcpp area max cloud optical thickness',
        'cloud_class': 'dominant cloud class within segment',
        'aod': 'aerosol optical depth',
        'tcwv': 'total column water vapour',
        'tco3': 'total column ozone'
    }

    units = {
        'duration': 's',
        'ghi_max_abs': 'W m-2',
        'ghi_mean_abs': 'W m-2',
        'ghi_max_rel': 'fraction',
        'ghi_mean_rel': 'fraction',
        'ce_sum': 'kJ m-2',
        'ce_max': 'W m-2',
        'dif_wrt_cs': '-',
        'dif_wrt_ghi': '-',
        'solar_angle': 'degrees',
        'dir_min': 'W m-2',
        'class_g1': '-',
        'u200': 'm s-1',
        'cth': 'm',
        'cot': '-',
        'cloud_class': '-',
        'aod': '-',
        'tcwv': 'mm',
        'tco3': 'DU'
    }

    # apply the attributes
    for variable in stats:
        dset[variable].attrs['long_name'] = long_names[variable]
        if variable != 'start_time':  # xarray already assigns the correct unit to datetime objects
            dset[variable].attrs['units'] = units[variable]

    return dset


def analyse_segments(data, segments, class_index, class_name, return_format='xarray', msgcpp=None):
    """
    Use the segments data to process and derive statistics

    :param data: irradiance dataset
    :param segments: segment analysis for given dataset
    :param class_index: index label associated with class name
    :param class_name: name of class (see netcdf attribute for names)
    :param str return_format: whether to return as xarray dataset or dictionary
    :param msgcpp: optional msgcpp timeseries
    :return:
    """

    # now, get some statistics from this
    stats = {
        'duration': [],
        'ghi_max_abs': [],
        'ghi_mean_abs': [],
        'ghi_max_rel': [],
        'ghi_mean_rel': [],
        'dif_wrt_cs': [],
        'dif_wrt_ghi': [],
        'start_time': [],
        'solar_angle': [],
        'dir_min': [],
        'class_g1': [],
        'u200': [],
        'cth': [],
        'cot': [],
        'cloud_class': [],
        'aod': [],
        'tcwv': [],
        'tco3': []
    }

    if class_name == 'cloud-enh.':
        stats['ce_sum'] = []
        stats['ce_max'] = []

    # intepolate clear-sky values to the irradiance measurement resolution
    if data.attrs['resolution'] == '1sec':
        data_ghi_cs = data.ghi_cs.interp(time_mcc=data.time_rad)
        data_u200 = data.wspd.sel(z=200).interp(time_tower=data.time_rad)
        data_msg = msgcpp.interp(datetime=data.time_rad, method='nearest') if msgcpp is not None else None
        data_mcc = data[['aod', 'tcwv', 'tco3']].interp(time_mcc=data.time_rad)
        factor = 1
        slice_dim = 'time_rad'
    else:
        warnings.warn("1 minute resolution segments not supported anymore.")
        return

    for segment in segments:
        if segment.mean() == class_index:
            # select subset from data based on segment time
            slicer = slice(segment.time_class[0], segment.time_class[-1])
            subset = data.sel(time_rad=slicer, time_class=slicer)
            subset_cs = data_ghi_cs.sel({slice_dim: slicer})
            subset_u200 = data_u200.sel(time_rad=slicer)
            subset_msg = data_msg.sel(time_rad=slicer) if data_msg is not None else None
            ghi = subset.ghi.values
            ghi_cs = subset_cs.values
            ghi_cs[ghi_cs < 1] = np.nan

            # calculate stats from segment
            stats['duration'].append(int((slicer.stop - slicer.start) / 1e9))
            stats['ghi_max_abs'].append(ghi.max())
            stats['ghi_mean_abs'].append(ghi.mean())
            stats['ghi_max_rel'].append((ghi / ghi_cs).max())
            stats['ghi_mean_rel'].append(ghi.mean() / ghi_cs.mean())
            stats['dif_wrt_cs'].append(subset.dif.mean() / ghi_cs.mean())
            stats['dif_wrt_ghi'].append((subset.dif.values / ghi).mean())
            stats['solar_angle'].append(subset.solar_elev.mean())
            stats['dir_min'].append(subset.dni.min())
            stats['class_g1'].append(np.argmax(np.bincount(subset.classes_group_1)))
            stats['u200'].append(subset_u200.mean())
            stats['aod'].append(data_mcc.aod.mean())
            stats['tcwv'].append(data_mcc.tcwv.mean())
            stats['tco3'].append(data_mcc.tco3.mean())

            if class_name == 'cloud-enh.':
                stats['ce_sum'].append((ghi - ghi_cs).sum() * factor / 1e3)
                stats['ce_max'].append((ghi - ghi_cs).max())

            if subset_msg is not None:
                stats['cth'].append(subset_msg.cth_max.mean(dim='time_rad'))
                stats['cot'].append(subset_msg.cot_max.mean(dim='time_rad'))
                dom_class = []
                for r in subset_msg.radius:
                    ssmsg = subset_msg.dom_class.sel(radius=r)
                    dom_class.append(np.argmax(np.bincount(ssmsg + 1)) - 1)
                stats['cloud_class'].append(dom_class)
            else:
                stats['cth'].append([np.nan]*len(preprocess_validation.msg_radii))
                stats['cot'].append([np.nan]*len(preprocess_validation.msg_radii))
                stats['cloud_class'].append([np.nan]*len(preprocess_validation.msg_radii))

            # add the starting time of the segment
            stats['start_time'].append(datetime.utcfromtimestamp(slicer.start.to_pandas().tolist() / 1e9))

    if return_format == 'xarray':
        return stats_dict_to_xarray(stats)
    else:
        return stats


def batch_run_segment_analyzer(dts, class_name, class_group, label='default', res='1min'):
    """

    :param list dts: list of datetimes of dates to process
    :param str class_name: name of the classification
    :param str class_group: name of the group the class belongs to
    :param str label: how to label this set of datetimes
    :param str res: which resolution dataset to pick (1min or 1sec)
    :return:
    """
    result = []
    daily_stats = dict(count=('date', []), duration=('date', []))

    for i, dt in enumerate(dts):
        # load data
        fpath = gutils.generate_processed_fpath(date=dt, res=res)
        if not os.path.isfile(fpath):
            daily_stats['count'][1].append(np.nan)
            daily_stats['duration'][1].append(np.nan)
            continue
        data = gutils.load_timeseries_data(fpath, apply_quality_control=True)
        classes = data[class_group]

        # load optional msgcpp data to relate to cloud type
        fpath_msg = gutils.generate_processed_fpath(date=dt, which='msgcpp')
        data_msg = xarray.open_dataset(fpath_msg) if os.path.isfile(fpath_msg) else None

        # analyse segment for date
        segments = find_segments(classes)
        segment_analysis = analyse_segments(data, segments, class_index=classes.class_names.index(class_name),
                                            class_name=class_name, return_format='dict', msgcpp=data_msg)
        result.append(segment_analysis)
        # count some stats per date
        daily_stats['count'][1].append(len(result[-1]['duration']))
        daily_stats['duration'][1].append(sum(result[-1]['duration']))

        print("Processed %s, added %s segments (%s/%s done)" % (dt.strftime('%Y%m%d'), len(result[-1]['duration']),
                                                                i + 1, len(dts)))

    # merge results together (using the first element as target)
    for result_ in result[1:]:
        for key in result_:
            result[0][key] += result_[key]

    # convert total stats to xarray and export
    dset = stats_dict_to_xarray(result[0])
    fdir_out = os.path.join('.', 'output')
    fpath_out = os.path.join(fdir_out, '%s_segments_total_stats_%s_%s.nc' % (res, label, class_name.replace('.', '')))
    if os.path.isfile(fpath_out):
        warnings.warn("Output file '%s' already exists, appending datetime to fname")
        fpath_out = fpath_out.replace('.nc', datetime.utcnow().strftime('_%Y%m%d_%H%M.nc'))
    dset.attrs['class'] = class_name
    dset.attrs['resolution'] = res
    dset.to_netcdf(fpath_out)

    # convert daily stats to xarray and export
    dset = xarray.Dataset(data_vars=daily_stats, coords=dict(date=dts))
    fpath_out = os.path.join(fdir_out, '%s_segments_daily_stats_%s_%s.nc' % (res, label, class_name.replace('.', '')))
    if os.path.isfile(fpath_out):
        warnings.warn("Output file '%s' already exists, appending datetime to fname")
        fpath_out = fpath_out.replace('.nc', datetime.utcnow().strftime('_%Y%m%d_%H%M.nc'))
    dset.attrs['resolution'] = res
    dset.attrs['class'] = class_name
    dset.to_netcdf(fpath_out)


if __name__ == "__main__":
    dts = gutils.generate_dt_range(datetime(2011, 2, 18), datetime(2021, 1, 1), delta_dt=timedelta(days=1))
    # dts = [datetime(2015, 4, 23)]

    # batch_run_segment_analyzer(dts, class_name='cloud-enh.', class_group='classes_group_2', label='all', res='1sec')
    batch_run_segment_analyzer(dts, class_name='shadow', class_group='classes_group_2', label='all', res='1sec')
    # batch_run_segment_analyzer(dts, class_name='variable', class_group='classes_group_1', label='all', res='1sec')
