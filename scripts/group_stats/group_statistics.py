import xarray
import os
from datetime import datetime, timedelta
import numpy as np
import warnings

from general import utils as gutils

fdir_out = './output/'
os.makedirs(fdir_out) if not os.path.exists(fdir_out) else None


def absolute_gradient_sum(x, axis=None):
    """
    calculate the sum of the absolute gradient, a straightforward proxy for daily variability
    does not account for daily cycle gradient, however.

    :param x:
    :param axis:
    :return:
    """
    gradient = np.nansum(np.abs(np.gradient(x)))
    return gradient


def variability_index(ghi, ghi_cs):
    """
    Calculate the variability index proposed by Reno et al. 2012 VI

    :param ghi:
    :param ghi_cs:
    :return: variability index dataset
    :rtype: xarray.Dataset
    """
    t_target = ghi.time_rad
    ghi_cs = ghi_cs.interp(time_mcc=t_target.values)
    delta_t = float((t_target[1] - t_target[0]).values) / 60e9  # delta_t in minutes

    ghi = ghi.values
    ghi_cs = ghi_cs.values
    ghi_cs[np.isnan(ghi)] = np.nan

    num = np.nansum(np.sqrt(np.diff(ghi) ** 2 + delta_t ** 2))
    den = np.nansum(np.sqrt(np.diff(ghi_cs) ** 2 + delta_t ** 2))

    return num / den


def cloud_modification_factor(ghi, ghi_cs, sea):
    """
    calculate the cloud modification factor while taking care of nan values and only positive solar elavation angles

    :param ghi: GHI data
    :param ghi_cs: GHI clearsky (mcclear) data
    :param sea: solar elevation angle
    :return:
    """
    t_target = ghi.time_rad
    ghi_cs = ghi_cs.interp(time_mcc=t_target.values)
    ghi_cs = ghi_cs.rename(dict(time_mcc='time_rad'))

    ghi = ghi.where(sea > 0., drop=True).values
    ghi_cs = ghi_cs.where(sea > 0., drop=True).values
    ghi_cs[np.isnan(ghi)] = 0.
    ghi[np.isnan(ghi)] = 0.

    if np.sum(ghi_cs) == 0:
        return np.nan
    else:
        return np.sum(ghi) / np.sum(ghi_cs)


def integrate(x, axis=None, floor=None):
    """
    Straightforward time integration of data, taking account of nan values and optionally adds a minimum value (floor)
    to data. This also requires some extra treatment due to nan values to prevent warnings.

    :param x:
    :param axis:
    :param floor: optional minimal value imposed on data
    :return:
    """
    if floor is not None:
        with np.errstate(invalid='ignore'):
            x[np.less(x, floor)] = floor
    realcount = np.sum(~np.isnan(x))
    if realcount == 0:
        return 0
    else:
        return np.nanmean(x) * realcount * (86400 / len(x))


def get_availability_fraction(data, mask=None):
    if mask is None:
        return np.sum(~np.isnan(data)) / len(data)
    else:
        return np.sum(~np.isnan(data[mask])) / np.sum(mask)


def classify_timeseries(ghi, ghi_cs, dif, dir, vi, fr_ghi, fr_dif, fr_dir, n, n_ce, n_sunny, n_shadow, n_cs,
                        group_size):
    """
    general classification of timeseries into broad categories:
    - overcast (no direct sunlight in subset)
    - clearsky (ghi==ghi_cs and low variability in subset)
    - cumulus (thresholds for amount of CE events, shadow/weak sunlight events and no clear-sky, upper limit on DIF_fr)
    - invalid (subsets that have invalid/incomplete data)
    - residual (anything that doesn't fall into one of the above categories)

    :param ghi: sum of ghi
    :param ghi_cs: sum of ghi_cs
    :param dif: sum of dif
    :param dir: sum of dir (beam) irradiance
    :param vi: variability index of ghi
    :param fr_ghi: availability fraction of GHI
    :param fr_dif: availability fraction of DIF
    :param n: number of measurement in cycle with positive solar angle
    :param n_ce: number of cloud enhancement events
    :param n_sunny: number of sunny events (wmo definition)
    :param n_shadow: number of shadow events
    :param n_cs: number of clear-sky events
    :param group_size: 'daily' or 'hourly'
    :return:
    """
    # set resolution and group size sensitive criteria
    if group_size == 'daily':
        cs_limit = None if n == 86400 else 1.2
        cu_limit = None if n == 86400 else 20
        oc_limit = 1e5
        ce_fr, ss_fr, cs_fr = 0.25, 0.15, 0.1
    else:
        cs_limit = None if n == 3600 else 1.2
        cu_limit = None if n == 3600 else 20
        oc_limit = 5e4
        ce_fr, ss_fr, cs_fr = 0.25, 0.15, 0

    # if some criteria are not set, stop the classification
    if None in [cs_limit, cu_limit, oc_limit]:
        warnings.warn('No criteria available for this res. and group size, skipping group classification.')
        return -1
    if n < 55:
        return -1  # not enough data points available (only occurs for hourly stats)

    # start classifying, assuming a 'residual' base result (result=0)
    result = 0
    if (dir / ghi_cs) < 0.01 and dir < oc_limit:
        result = 1  # overcast
    elif (ghi / fr_ghi) / ghi_cs > 0.96 and vi < cs_limit:
        result = 2  # clear sky
    elif ((n_ce / n) >= ce_fr) and ((n_shadow + n_sunny) / n >= ss_fr) and (n_cs / n <= cs_fr):
        if (dif / ghi_cs < 0.45) and vi > cu_limit:
            result = 3  # cumulus

    # check result validity
    if result == 1 and fr_dir > 0.95:
        return result  # overcast result is valid
    elif result == 2 and fr_ghi > 0.95:
        return result  # clear sky result in valid
    elif min(fr_dir, fr_dif, fr_ghi) > 0.95:
        return result  # residual or cumulus results are valid
    else:
        return -1  # invalid result, too much missing data to safely classify


attrs = {
    # aggregate stats
    'n_possea': dict(long_name='Number of positive solar elevation angles', units='-'),
    'm_sea': dict(long_name='Mean solar elevation angle', units='degrees'),
    'sag_ghi': dict(long_name='Sum of absolute gradient of global horizontal irradiance', units='J/m2'),
    'sag_dir': dict(long_name='Sum of absolute gradient of short wave beam irradiance', units='J/m2'),
    's_ghi': dict(long_name='Sum of global horizontal irradiance', units='J/m2'),
    's_ghi_cs': dict(long_name='Sum of clear-sky global horizontal irradiance', units='J/m2'),
    's_dif': dict(long_name='Sum of diffuse irradiance', units='J/m2'),
    's_dir': dict(long_name='Sum of short wave beam irradiance', units='J/m2'),
    's_dhi_gmd': dict(long_name='Sum of short wave direct horizontal (ghi-dif) irradiance', units='J/m2'),
    's_dhi_sac': dict(long_name='Sum of short wave direct horizontal (solar-angle corrected) irradiance', units='J/m2'),
    's_lwd': dict(long_name='Sum of long wave down', units='J/m2'),
    's_lwd_day': dict(long_name='Sum of long wave down during daytime', units='J/m2'),

    # indices
    'vi_ghi': dict(long_name='Variability index based on GHI', units='-'),
    'vi_dif': dict(long_name='Variability index based on DIF', units='-'),
    'cmf': dict(long_name='Cloud modification factor', units='-'),

    # classification stats
    'n_residual_g1': dict(long_name='Number of residuals in group 1', units='-'),
    'n_na_g1': dict(long_name='Number of n/a in group 1', units='-'),
    'n_residual_g2': dict(long_name='Number of residuals in group 2', units='-'),
    'n_na_g2': dict(long_name='Number of n/a in group 2', units='-'),
    'n_clearsky': dict(long_name='Number of clear sky measurements', units='-'),
    'n_overcast': dict(long_name='Number of overcast measurements', units='-'),
    'n_variable': dict(long_name='Number of variable measurements', units='-'),
    'n_shadow': dict(long_name='Number of shadow measurements', units='-'),
    'n_sunshine': dict(long_name='Number of sunny measurements', units='-'),
    'n_ce': dict(long_name='Number of cloud enhancement measurements', units='-'),

    # misc
    'n': dict(long_name='Number of measurements', units='-'),
    'fr_ghi': dict(long_name='Fraction of available GHI measurements during daytime', units='-'),
    'fr_dir': dict(long_name='Fraction of available DIF measurements during daytime', units='-'),
    'fr_dif': dict(long_name='Fraction of available DIR measurements during daytime', units='-'),
    'fr_all': dict(long_name='Fraction of availability of all three components (GHI,DIR,DIF) during daytime', units='-')
}


def group_data_into_stats(dates, group_size='daily', source_data='bsrn', source_res='1min'):
    """
    aggregate irradiance measurements and derived properties into various time frames

    :param dates: list of dates (datetimes) to process
    :param str group_size: preset size of groups the data is processed in ('daily', 'hourly')
    :param str source_data: irradiance data source (only supports BSRN for now)
    :param str source_res: resolution version of the source data
    :return:
    """

    # predefined location to save results in
    results = {i: [] for i in attrs}
    datetimes_axis = []
    skip_classes = False
    dates = sorted(dates)

    # run analysis
    for i, dt in enumerate(dates):
        # load dataset
        fpath = gutils.generate_processed_fpath(dt, res=source_res)
        if not os.path.isfile(fpath):
            for resvar in results:
                if resvar in ['n', 'fr_ghi', 'fr_dif', 'fr_dir', 'fr_all']:
                    results[resvar].append(0)
                else:
                    results[resvar].append(np.nan)
            datetimes_axis.append(dt)
            continue
        else:
            base_data = gutils.load_timeseries_data(fpath, apply_quality_control=True)

        if i % 10 == 0:
            print('[%s/%s] Processing aggregate stats for %s' % (i + 1, len(dts), dt))

        # generate datetime selection based on requests group size
        if group_size == 'daily':
            datetimes_axis.append(dt)
            times = [dt]
        elif group_size == 'hourly':
            times = gutils.generate_dt_range(dt, dt + timedelta(days=1), delta_dt=timedelta(hours=1))
            datetimes_axis += times
        else:
            return

        for time in times:
            if len(times) == 1:
                data = base_data
            else:
                slicer = slice(time, time + timedelta(hours=1, minutes=-1))
                data = base_data.sel(time_rad=slicer, time_mcc=slicer, time_class=slicer)

            # aggregate stats
            results['n_possea'].append(np.sum(data.solar_elev > 0))
            results['m_sea'].append(np.mean(data.solar_elev))
            results['sag_ghi'].append(data.ghi.reduce(absolute_gradient_sum))
            results['sag_dir'].append(data.dni.reduce(absolute_gradient_sum))
            results['s_ghi'].append(data.ghi.reduce(integrate, floor=0.))
            results['s_dif'].append(data.dif.reduce(integrate, floor=0.))
            results['s_dir'].append(integrate(data.dni.where((data.solar_elev > 0) & (data.dni > 2))))
            results['s_dhi_gmd'].append(data.dhi_gmd.reduce(integrate))
            results['s_dhi_sac'].append(integrate(data.dhi_sac.where((data.solar_elev > 0) & (data.dni > 2))))
            results['s_ghi_cs'].append(data.ghi_cs.reduce(integrate))
            results['s_lwd'].append(data.lwd.reduce(integrate))
            results['s_lwd_day'].append(integrate(data.lwd.where(data.solar_elev > 0)))

            # indices
            results['vi_ghi'].append(variability_index(data.ghi, data.ghi_cs))
            results['vi_dif'].append(variability_index(data.dif, data.dif_cs))
            results['cmf'].append(cloud_modification_factor(data.ghi, data.ghi_cs, data.solar_elev))

            # classification stats
            if 'classes_group_1' in data:
                results['n_residual_g1'].append(np.sum(data.classes_group_1 == 0))
                results['n_na_g1'].append(np.sum((data.classes_group_1.values == 1) & (data.solar_elev.values > 0)))
                results['n_clearsky'].append(np.sum(data.classes_group_1 == 2))
                results['n_overcast'].append(np.sum(data.classes_group_1 == 3))
                results['n_variable'].append(np.sum(data.classes_group_1 == 4))
            else:
                skip_classes = True

            if 'classes_group_2' in data and not skip_classes:
                results['n_residual_g2'].append(np.sum(data.classes_group_2 == 0))
                results['n_na_g2'].append(np.sum((data.classes_group_2.values == 1) & (data.solar_elev.values > 0)))
                results['n_shadow'].append(np.sum(data.classes_group_2 == 2))
                results['n_sunshine'].append(np.sum(data.classes_group_2 == 3))
                results['n_ce'].append(np.sum(data.classes_group_2 == 4))
            else:
                skip_classes = True

            # misc
            results['n'].append(len(data.time_rad))
            results['fr_ghi'].append(get_availability_fraction(data.ghi, mask=data.solar_elev > 0.))
            results['fr_dir'].append(get_availability_fraction(data.dni, mask=data.solar_elev > 0.))
            results['fr_dif'].append(get_availability_fraction(data.dif, mask=data.solar_elev > 0.))
            results['fr_all'].append(get_availability_fraction(data.ghi + data.dni + data.dif,
                                                               mask=data.solar_elev > 0.))

    # create a dataset for results
    xr_arrays = {}

    for var, result in results.items():
        if len(result) == len(datetimes_axis):
            xr_arrays[var] = ('date', result)
        elif skip_classes:
            warnings.warn("Classification statistics are skipped, because one or more datasets are incomplete")
        else:
            warnings.warn(
                "Amount of points for '%s' are not equal to all dts (%s/%s), adding anyway" % (var, len(results),
                                                                                               len(dts)))
            xr_arrays[var] = ('date', result)

    ds_stats = xarray.Dataset(data_vars=xr_arrays, coords={'date': datetimes_axis})
    ds_stats.attrs['group_size'] = group_size
    ds_stats.attrs['source_resolution'] = source_res

    # add attributes to the data arrays
    for var in ds_stats:
        ds_stats[var].attrs = attrs[var]

    # create output path
    fpath = os.path.join(fdir_out, '%s_stats_%s_%s.nc' % (group_size, source_data, source_res))
    if os.path.isfile(fpath):
        warnings.warn("Output file already exists, appending current datetime to filename")
        fpath = fpath.replace('.nc', datetime.utcnow().strftime('_%Y%m%d-%H%M.nc'))

    # export the data
    ds_stats.to_netcdf(fpath)


if __name__ == "__main__":
    # set range to process
    dts = []
    # for year in range(2006, 2020):
    #     dts += gutils.generate_dt_range(datetime(year, 1, 1), datetime(year+1, 1, 1), delta_dt=timedelta(days=1))
    dts += gutils.generate_dt_range(datetime(2011, 1, 1), datetime(2021, 1, 1), delta_dt=timedelta(days=1))
    # dts += gutils.generate_dt_range(datetime(2018, 8, 1), datetime(2018, 9, 1), delta_dt=timedelta(days=1))

    group_data_into_stats(dts, group_size='daily', source_data='bsrn', source_res='1sec')
