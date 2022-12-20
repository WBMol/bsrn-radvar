import xarray
import os
from datetime import datetime
import numpy as np
import pandas as pd

from general import utils as gutils
import settings


def verify_classification(variable, rad_classes, ref_data, reference):
    """
    Verify a variable of radiation data using msg-cpp satellite data, nubiscope or a combined validation product

    :param variable: variable to verify, needs to be supported
    :param rad_classes: radiation classification data
    :param ref_data: MSGCPP or Nubiscope data
    :param str reference: reference dataset type
    :return:
    """
    # select valid datapoints from reference data
    if reference == 'msgcpp':
        ref_data = ref_data.where(ref_data.ccover >= 0, drop=True)
    elif reference == 'nubiscope':
        ref_data = ref_data[variable]
    elif reference == 'class_validation':
        ref_data = ref_data[variable]
        ref_data = xarray.where(ref_data <= 1, ref_data, np.nan)  # mask numbers 2&3; they are disagreement flags
    else:
        raise NotImplementedError

    # select subset from outer temporal ranges
    dt_subset = slice(ref_data.datetime[0], ref_data.datetime[-1])
    rad_classes = rad_classes.sel(time_class=dt_subset)

    # save class attributes to later use
    class_attrs = rad_classes.attrs

    # align the satellite data with radiation
    ref_data = ref_data.interp(datetime=rad_classes.time_class, method='nearest')

    # mark data that is no possible to classify due to missing data with nans
    rad_classes = xarray.where(rad_classes != 1, rad_classes, np.nan)

    # mask out mutual nans so only valid data is left
    if reference == 'msgcpp':
        nan_mask = ref_data.ccover.isnull() | rad_classes.isnull()
        ccover = xarray.where(~nan_mask.values, ref_data.ccover, np.nan).dropna(dim='time_class')
        rad_classes = xarray.where(~nan_mask, rad_classes, np.nan).dropna(dim='time_class')

        if variable == 'clearsky':
            # select clearsky data from radiation and satellite
            rad = rad_classes == class_attrs['class_names'].index('clearsky')
            ref = (ccover < 0.05).values
            rad = rad.values
        elif variable == 'overcast':
            # select overcast data and interpolate cloud cover to radiation
            rad = rad_classes == class_attrs['class_names'].index('overcast')
            cot = xarray.where(~nan_mask.values, ref_data.cot, np.nan).dropna(dim='time_class')
            ref = ((ccover > 0.95) & (cot > 3.6)).values
            rad = rad.values
        else:
            print("Classification of '%s' not supported" % variable)
            return {}
    elif reference in ['nubiscope', 'class_validation']:
        # create mutual mask
        nan_mask = ref_data.isnull() | rad_classes.isnull()

        # extract from radiation data
        rad_classes = xarray.where(~nan_mask, rad_classes, np.nan).dropna(dim='time_class')
        rad = rad_classes == class_attrs['class_names'].index(variable)
        rad = rad.values

        # extract from reference data
        ref = xarray.where(~nan_mask.values, ref_data, np.nan).dropna(dim='time_class')
        ref = ref == 1
    else:
        raise NotImplementedError("Reference dataset not supported: %s" % reference)

    # calculate counts for basic stats
    result = {
        'tp': (rad & ref).sum(),
        'fp': ((rad == True) & (ref == False)).sum(),
        'fn': ((rad == False) & (ref == True)).sum(),
        'tn': (~rad & ~ref).sum(),
        'n': len(rad)
    }
    return result


def convert_results_to_xarray(results, dates):
    """

    :param results:
    :return:
    :rtype: xarray.Dataset
    """
    # first concat the results along the score direction
    new_results = {}
    for result in results:
        for variable in result:
            if variable not in new_results:
                new_results[variable] = {}
            for score in result[variable]:
                if score not in new_results[variable]:
                    new_results[variable][score] = []
                new_results[variable][score].append(result[variable][score])

    # return into an easy to use xarray dataset
    results_ds = {}
    for variable in new_results:
        das = [xarray.DataArray(new_results[variable][var], name=var, dims=('date',), coords=dict(date=dates))
               for var in new_results[variable]]
        results_ds[variable] = xarray.merge(das)
    dset = xarray.concat(results_ds.values(), dim='classification')
    dset['classification'] = list(results_ds.keys())

    # add attributes
    dset['fn'].attrs['long_name'] = 'False negatives'
    dset['fp'].attrs['long_name'] = 'False positives'
    dset['tn'].attrs['long_name'] = 'True positives'
    dset['tp'].attrs['long_name'] = 'True negatives'
    dset['n'].attrs['long_name'] = 'Total sample size'
    return dset


def calculate_skill_scores(res='1sec', reference='msgcpp'):
    """
    calculate basic skill scores based on contingency table statistics

    :param str res: dataset resolution
    :param str reference: reference validation dataset
    :return:
    """
    # specify long names of statistics
    long_names = dict(acc='Accuracy', pacc='Accuracy of positives', bias='Positive detection bias',
                      pod='Probability of detection', far='False alarm ratio', gss='Gilbert skill score',
                      tp='True positive', tn='True negative', fp='False positive', fn='False negative')

    # load the dataset with contingency table counts
    stats = xarray.open_dataset(os.path.join(settings.fdir_data_out, 'class_validation_%s_%s.nc' % (reference, res)))

    # first, calculate stuff per month, then for the whole dataset
    periods = ['year', 'month', 'day']

    for period in periods:
        if period == 'year':
            stats_ = stats.groupby('date.year').sum()
        elif period == 'month':
            stats_ = stats.groupby('date.month').sum()
        elif period == 'all':
            stats_ = stats.sum(dim='date')
        else:
            stats_ = stats

        # calculate skill scores
        acc = (stats_.tn + stats_.tp) / stats_.n
        pacc = stats_.tp / (stats_.tp + stats_.fp)
        bias = (stats_.tp + stats_.fp) / (stats_.tp + stats_.fn)
        pod = stats_.tp / (stats_.tp + stats_.fn)
        far = stats_.fp / (stats_.tp + stats_.fp)
        tp_rnd = (stats_.tp + stats_.fp) * (stats_.tp + stats_.fn) / stats_.n
        gss = (stats_.tp - tp_rnd) / (stats_.tp + stats_.fn + stats_.fp - tp_rnd)

        # calculate relative hits etc
        tp = stats_.tp / stats_.n
        tn = stats_.tn / stats_.n
        fp = stats_.fp / stats_.n
        fn = stats_.fn / stats_.n

        dset = xarray.Dataset(data_vars=dict(acc=acc, pacc=pacc, bias=bias, pod=pod, far=far, gss=gss,
                                             tp=tp, tn=tn, fp=fp, fn=fn, n=stats_.n))
        for v in long_names:
            dset[v].attrs['long_name'] = long_names[v]
        dset.to_netcdf(os.path.join(settings.fdir_data_out, 'skill_scores_%s_%s_%s.nc' % (reference, res, period)))


def visualize_skill_scores(res='1sec', reference='msgcpp'):
    """
    visualize skill scores

    :param str res: resolution of radiation dataset
    :param str reference: reference dataset used for validation
    :return:
    """
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 100)
    # pd.set_option('display.precision', 3)
    pd.set_option('display.float_format', '{:.3f}'.format)

    for period in ['year', 'month', 'day']:
        stats = xarray.open_dataset(os.path.join(settings.fdir_data_out, 'skill_scores_%s_%s_%s.nc' %
                                                 (reference, res, period)))
        result_str = ''

        for c in stats.classification:
            stats_ = stats.sel(classification=c).drop_vars(['classification']).to_dataframe()
            result_str += '%s\n\n' % str(c.values)
            result_str += str(stats_)
            result_str += '\n\n---------------------------------------\n\n'

        with open(os.path.join(settings.fdir_data_out, 'skill_score_%s_%s_%s.txt' % (reference, res, period)),
                  'w') as f:
            f.write(result_str)


def run_statistical_validation(dates, res='1sec', reference='msgcpp'):
    """
    batch plot multiple datetimes for which classifications are available

    :param list dates: list of dates (datetimes) to process
    :param str res: source resolution dataset (1min or 1sec)
    :param str reference: the validation dataset, msgcpp or nubiscope
    :return:
    """
    results = []

    for dt in dates:
        fpath_bsrn = gutils.generate_processed_fpath(dt, res=res, which='bsrn')
        fpath_ref = gutils.generate_processed_fpath(dt, which=reference)

        if os.path.isfile(fpath_bsrn) & os.path.isfile(fpath_ref):
            try:
                data = gutils.load_timeseries_data(fpath_bsrn)['classes_group_1']
                data_ref = xarray.open_dataset(fpath_ref)

                result = {}
                for variable in ['clearsky', 'overcast']:
                    result[variable] = verify_classification(variable, data, data_ref, reference=reference)
                results.append(result)

            except KeyError as e:
                print('Failed to load classification data for %s (%s)' % (dt, e))
                result = {i: 0 for i in ['tp', 'fp', 'fn', 'tn', 'n']}
                results.append({i: result for i in ['clearsky', 'overcast']})
        else:
            print("Input files missing for date: %s" % dt)
            result = {i: 0 for i in ['tp', 'fp', 'fn', 'tn', 'n']}
            results.append({i: result for i in ['clearsky', 'overcast']})

        print("[%i/%i] Done: %s" % (dates.index(dt) + 1, len(dates), dt))

    # export the classification validation to a xarray dataset
    results = convert_results_to_xarray(results, dates)
    results.to_netcdf(os.path.join(settings.fdir_data_out, 'class_validation_%s_%s.nc' % (reference, res)))


if __name__ == "__main__":
    dts = gutils.generate_dt_range(datetime(2014, 1, 1), datetime(2017, 1, 1))
    # dts = [datetime(2016, 6, 16)]
    ref = 'class_validation'

    run_statistical_validation(dates=dts, res='1sec', reference=ref)
    calculate_skill_scores(reference=ref)
    visualize_skill_scores(reference=ref)
