import xarray
import os
from datetime import datetime, timedelta, timezone
import pandas as pd
import numpy as np
import pysolar.solar as solar
import warnings

from scripts.processor import settings
from scripts.processor import utils


@utils.timeit
def preprocess_data(time_start, time_stop, res='1sec'):
    """
    Process all raw data within a datetime range, grouped per day

    :param datetime.datetime time_start: start of the datetime range to process
    :param datetime.datetime time_stop: end of the datetime range
    :param str res: '1sec' or '1min', though source data is usually '1sec'
    :return:
    """
    # check that the provided arguments are valid and supported
    assert time_start.timestamp() % 600 == 0, "Start of time range should be in 10s of minutes!"
    assert (time_stop - time_start).total_seconds() >= 3600, "Datetime range should be at least one hour!"

    # split the full requested timerange to process into separate days
    day_ranges = utils.split_dtrange_into_ranges(time_start, time_stop, ranges='days')

    # further process the raw data
    for time_start_, time_stop_ in day_ranges:
        add_derived_variables(dt=time_start_, res=res)

    # add mcclear clear-sky @ 1 min (separate loop for file reading efficiency)
    add_clear_sky_ghi(time_start, time_stop, res=res)

    # add mcclear AOD @ 1 hour
    add_mcclear_atmos_info(time_start, time_stop, res=res)

    # add Cabauw tower profile
    add_cabauw_tower_profiles(time_start, time_stop, res=res)

    for time_start_, time_stop_ in day_ranges:
        quality_filter(dt=time_start_)

    # add_lcl_and_path(time_start, time_stop, res=res)
    post_processing_cleanup(time_start, time_stop, update_version=False)


@utils.timeit
def specific_quality_check_routine(subset):
    """
    Run quality filter for a specific set of preprocessed dates

    :param subset: if 'test' then it'll run the test cases in debug mode, otherwise subset = (date start, date stop)
    :return:
    """
    if subset == 'test':
        from qc_cases import cases
        for case_type in cases:
            for date in cases[case_type]:
                quality_filter(dt=date, debug=True)
    elif type(subset) == tuple:
        day_ranges = utils.split_dtrange_into_ranges(subset[0], subset[1], ranges='days')
        for day in day_ranges:
            quality_filter(dt=day[0], debug=False)
    elif type(subset) == datetime:
        quality_filter(dt=subset, debug=True)
    else:
        raise NotImplementedError('subset isnt supported: %s' % subset)


@utils.timeit
def add_derived_variables(dt, res):
    """
    use the preprocessed, merged netcdf file of variables and calculate derived variables from the core set of data

    :param datetime.datetime dt: day to process
    :param str res: which dataset resolution to add to ('1min' or '1sec')
    :return:
    """
    # set up file dir and paths
    fdir = settings.fdir_out.format(res=res, y=dt.year, m=dt.month)
    fname = dt.strftime('%Y%m%d.nc')
    fpath = os.path.join(fdir, fname)
    fpath_tmp = fpath.replace('.nc', '.nc.tmp')

    # load dataset
    try:
        dset = xarray.open_dataset(fpath)
    except FileNotFoundError:
        print("File not found, skipping: %s" % fpath)
        return

    # check whether derived variables have already been added, and skip if so
    if 'solar_elev' in dset:
        print("Derived variables already added, skipping this function for: %s" % fpath)
        return

    # add timezone information to the datetime, required by PySolar
    dt = dt.replace(tzinfo=timezone.utc, hour=0, minute=0)

    # calculate the swd direct normal (glob-diff)
    v_glob = settings.bsrn_vars['shortwave_global']
    v_diff = settings.bsrn_vars['shortwave_diffuse']
    v_target = settings.bsrn_vars['shortwave_direct_horizontal_gmd']
    dset[v_target['name']] = dset[v_glob['name']] - dset[v_diff['name']]
    dset[v_target['name']].attrs['long_name'] = v_target['long_name']
    dset[v_target['name']].attrs['units'] = v_target['units']

    # calculate the swd direct normal (solar angle correction)
    v_dir = settings.bsrn_vars['shortwave_direct']
    v_target = settings.bsrn_vars['shortwave_direct_horizontal_sac']
    s_dir = dset[v_dir['name']]

    lat, lon = dset.location  # position of instrument

    if res == '1sec':
        delta_t = 1
    elif res == '1min':
        delta_t = 60
        if settings.solar_elevation_setting == 'max_accuracy':
            warnings.warn("Requested max_accuracy for solar angle, but exceeds data res. Falling back to 'accurate'")
            settings.solar_elevation_setting = 'accurate'
    else:
        # assuming 1 sec by default, but if code should raise a warning here: res is ill-defined
        delta_t = 1
        warnings.warn("'res' is set to an unknown value of '%s', defaulting to 1-sec resolution" % res)

    soc = np.arange(0, 86400, delta_t)  # second of day
    dts = [dt + timedelta(seconds=int(i)) for i in soc]

    if settings.solar_elevation_setting == 'fast':
        angle_res = 60 // delta_t
        angles = [solar.get_altitude_fast(lat, lon, when=dt) for dt in dts[::angle_res]]
    elif settings.solar_elevation_setting == 'accurate':
        angle_res = 60 // delta_t
        angles = [solar.get_altitude(lat, lon, when=dt) for dt in dts[::angle_res]]
    elif settings.solar_elevation_setting == 'max_accuracy':
        angle_res = 2 if delta_t == 1 else 60
        angles = [solar.get_altitude(lat, lon, when=dt) for dt in dts[::angle_res]]
    else:
        raise Exception("Solar elevation option is not set to a valid value, check settings file")

    # get azimuth angles too, but no need for accuracy in any case yet
    azims = [solar.get_azimuth_fast(lat, lon, when=dt) for dt in dts[::300 // delta_t]]

    angles_interpolated = np.interp(soc, soc[::angle_res], angles)
    azims_interpolated = np.interp(soc, soc[::300 // delta_t], azims)

    dset[v_target['name']] = (s_dir * np.sin(np.deg2rad(angles_interpolated))).astype(np.float32)
    dset[v_target['name']].attrs['long_name'] = v_target['long_name']
    dset[v_target['name']].attrs['units'] = v_target['units']

    # add the solar angle as well
    time_rad_attrs = dset.time_rad.attrs
    da_angles = xarray.DataArray(angles_interpolated, dims={'time_rad': dts})
    dset['solar_elev'] = da_angles.astype(np.float32)
    dset['solar_elev'].attrs['long_name'] = 'Solar elevation angle'
    dset['solar_elev'].attrs['units'] = 'degrees'
    dset['solar_elev'].attrs['accuracy'] = settings.solar_elevation_setting
    da_azims = xarray.DataArray(azims_interpolated, dims={'time_rad': dts})
    dset['solar_azim'] = da_azims.astype(np.float32)
    dset['solar_azim'].attrs['long_name'] = 'Solar azimuth angle'
    dset['solar_azim'].attrs['units'] = 'degrees'
    dset['solar_azim'].attrs['accuracy'] = 'fast'
    dset['time_rad'].attrs = time_rad_attrs

    # write to file
    dset.to_netcdf(fpath_tmp)
    os.rename(fpath_tmp, fpath)


@utils.timeit
def add_clear_sky_ghi(time_start, time_stop, res, v='3.5'):
    """
    add the clear sky global horizontal irradiance from the CAMS McClear product

    :param time_start: start of the datetime range to process
    :param time_stop: end (not inclusive) of the datetime range to process
    :param str res: which dataset resolution to add to (1min or 1sec)
    :param str v: version of cams mcclear to use, set to the latest by default
    :return:
    """
    # load the mcclear dataset
    mc_dir = os.path.join(settings.fdir_mcc, 'Cabauw')
    mc_fpath = os.path.join(mc_dir, 'mcclear_20040101-20220101_v%s.nc' % v)
    mcclear = xarray.open_dataset(mc_fpath).squeeze().drop_vars(['altitude', 'latitude', 'longitude'])
    var_attrs = settings.mcclear_vars
    mc_vars = ['clear_sky_ghi', 'clear_sky_dhi']

    dt_range = utils.split_dtrange_into_ranges(time_start, time_stop, ranges='days')

    for time_start_, time_stop_ in dt_range:
        # load preprocessed dataset
        fdir = settings.fdir_out.format(res=res, y=time_start_.year, m=time_start_.month)
        fname = time_start_.strftime('%Y%m%d.nc')
        fpath = os.path.join(fdir, fname)
        fpath_tmp = fpath.replace('.nc', '.nc.tmp')

        try:
            data = xarray.open_dataset(fpath)
        except FileNotFoundError:
            print("File not found, skipping: %s" % fpath)
            continue

        # select a datetime range
        subset = mcclear.sel(time=slice(time_start_, time_stop_))
        das = [subset[var_attrs[var]['raw_name']] for var in mc_vars]
        new_dim = var_attrs[mc_vars[0]]['dim']  # should be identical for both vars, so just pick the first

        # rename the variables, dimensions for all DataArrays
        das = [da.rename({var_attrs[var]['raw_dim']: new_dim}).rename(var_attrs[var]['name']) for
               (var, da) in zip(mc_vars, das)]

        # reindex to full day
        dt = time_start_.replace(hour=0, minute=0)

        for var, da in zip(mc_vars, das):
            time_mcc = pd.date_range(dt, dt + timedelta(days=1),
                                     freq=settings.sample_frequency[var_attrs[var]['dim']])[:-1].to_numpy()
            da = da.reindex({new_dim: time_mcc})

            # convert units from 'Wh/m2' to 'W/m2' (Wh of 1 minute -> W:  Wh * 3600 / 60 = W)
            da = da * 60.

            # add attributes
            da.attrs['units'] = var_attrs[var]['units']
            da.attrs['long_name'] = var_attrs[var]['long_name']

            # merge
            if var_attrs[var]['name'] in data:
                data = data.drop_vars([var_attrs[var]['name']])  # drop if it already exists
            data = data.merge(da)

        # add temporal axis attribute
        data[new_dim].attrs['long_name'] = 'Datetime axis of mcclear clear-sky data'

        # export
        data.to_netcdf(fpath_tmp)
        os.rename(fpath_tmp, fpath)


@utils.timeit
def add_mcclear_atmos_info(time_start, time_stop, res, v='3.5'):
    """
    add the total AOD from the CAMS McClear product

    :param time_start: start of the datetime range to process
    :param time_stop: end (not inclusive) of the datetime range to process
    :param str res: which dataset resolution to add to (1min or 1sec)
    :param str v: the version to use, set to the latest by default
    :return:
    """
    # load the mcclear aod dataset
    mccai_fdir = os.path.join(settings.fdir_mcc, 'Cabauw')
    mccai_fname = 'mcclear_ai_2004-2022_v%s.nc' % v
    mccai_fpath = os.path.join(mccai_fdir, mccai_fname)
    mccai = xarray.open_dataset(mccai_fpath).squeeze()
    var_attrs = settings.mcclear_vars

    dt_range = utils.split_dtrange_into_ranges(time_start, time_stop, ranges='days')

    for time_start_, time_stop_ in dt_range:
        # load preprocessed dataset
        fdir = settings.fdir_out.format(res=res, y=time_start_.year, m=time_start_.month)
        fname = time_start_.strftime('%Y%m%d.nc')
        fpath = os.path.join(fdir, fname)
        fpath_tmp = fpath.replace('.nc', '.nc.tmp')

        try:
            data = xarray.open_dataset(fpath)
        except FileNotFoundError:
            print("File not found, skipping: %s" % fpath)
            continue

        # create time axis
        dt = time_start_.replace(hour=0, minute=0)
        time_mcc = pd.date_range(dt, dt + timedelta(days=1),
                                 freq=settings.sample_frequency[var_attrs['aod']['dim']])[:-1].to_numpy()

        # select a datetime range
        for mccvar in ['aod', 'tcwv', 'tco3']:
            subset = mccai.sel({var_attrs[mccvar]['raw_dim']: slice(time_start_, time_stop_)})
            da = subset[var_attrs[mccvar]['raw_name']]

            # rename the variables, dimensions for all DataArrays
            new_dim = var_attrs[mccvar]['dim']
            da = da.rename({var_attrs[mccvar]['raw_dim']: new_dim}).rename(var_attrs[mccvar]['name'])

            # reindex to full day
            da = da.reindex({new_dim: time_mcc})

            # add attributes
            da.attrs['units'] = var_attrs[mccvar]['units']
            da.attrs['long_name'] = var_attrs[mccvar]['long_name']
            da.attrs['description'] = var_attrs[mccvar]['description']

            # merge and add label to temporal axis
            if mccvar in data:
                data = data.drop_vars([mccvar])
            data = data.merge(da)

        data[new_dim].attrs['long_name'] = 'Datetime axis of McClear data'

        # export
        data.to_netcdf(fpath_tmp)
        os.rename(fpath_tmp, fpath)


@utils.timeit
def add_cabauw_tower_profiles(time_start, time_stop, res):
    """
    Add wind profiles of the cabauw tower to the dataset

    :param time_start:
    :param time_stop:
    :param res:
    :return:
    """
    fname_fmt = 'cesar_tower_meteo_lc1_t10_v1.0_%Y%m.nc'
    mon_ranges = utils.split_dtrange_into_ranges(time_start, time_stop, ranges='months')

    for time_start_, time_stop_ in mon_ranges:
        # load tower data, which is archived in monthly datasets
        fdir_tower = settings.fdir_ctower.format(y=time_start_.year)
        fpath_tower = os.path.join(fdir_tower, time_start_.strftime(fname_fmt))
        if not os.path.exists(fpath_tower):
            warnings.warn('Tower data does not exist for date: %s, skipping' % time_start_.strftime('%Y%m%d'))
            continue
        dset_tower = xarray.open_dataset(os.path.join(fdir_tower, fpath_tower))

        # preprocessing / simplification of tower data
        dset_tower['F'].attrs = dict(units='m s-1', long_name='Wind speed', description='Cabauw Tower Profile')
        dset_tower['D'].attrs = dict(units='deg', long_name='Wind direction', description='Cabauw Tower Profile')
        dset_tower['z'].attrs = dict(units='m', long_name='Height above ground', description='Cabauw Tower Profile')
        dset_tower = dset_tower.dropna(dim='z')

        # split month range into days
        day_ranges = utils.split_dtrange_into_ranges(time_start_, time_stop_, ranges='days')

        # fix the datetime axis by rounding of to nearest second (there are subsecond rounding errors..)
        dset_tower['time'] = dset_tower['time'].to_dataframe()['time'].dt.round('1s').to_xarray()

        # process each day
        for dt_1, dt_2 in day_ranges:
            # load preprocessed radiation dataset
            fdir = settings.fdir_out.format(res=res, y=dt_1.year, m=dt_1.month)
            fname = dt_1.strftime('%Y%m%d.nc')
            fpath = os.path.join(fdir, fname)
            fpath_tmp = fpath.replace('.nc', '.nc.tmp')

            try:
                data = xarray.open_dataset(fpath)
            except FileNotFoundError:
                print("File not found, skipping: %s" % fpath)
                continue

            # select subset from tower data
            tower_subset = dset_tower.sel(time=slice(dt_1, dt_2 - timedelta(minutes=1)))
            tower_subset = tower_subset.rename_dims({'time': 'time_tower'}).rename_vars({'time': 'time_tower'})
            tower_subset['time_tower'].attrs['long_name'] = 'Datetime axis of Cabauw tower measurements'
            tower_subset = tower_subset.rename({'F': 'wspd', 'D': 'wdir'})

            if 'wspd' in data:
                data = data.drop_vars(['wspd', 'wdir'])

            # add certain variables to the dataset
            data = data.merge(tower_subset['wspd'])
            data = data.merge(tower_subset['wdir'])

            # export
            data.to_netcdf(fpath_tmp)
            os.rename(fpath_tmp, fpath)


@utils.timeit
def quality_filter(dt, debug=False):
    """
    Remove 'unphysical' measurements from the timeseries by comparing local variability with general variability

    Most of these 'events' that need to be filtered are spontaneous strong dips or oscillations in the signal
    that last only briefly (10-20 seconds)

    :param dt:
    :param bool debug: whether to export extra information about the quality filter
    :return:
    """
    # generate filepaths
    fdir = settings.fdir_out.format(res='1sec', y=dt.year, m=dt.month)
    fname = dt.strftime('%Y%m%d.nc')
    fpath = os.path.join(fdir, fname)
    fpath_tmp = fpath.replace('.nc', '_tmp.nc')

    # load data
    if os.path.isfile(fpath):
        data = xarray.open_dataset(fpath)
    else:
        warnings.warn("Input file not found, skipping (%s)" % fpath)
        return

    # calculate which, if any, timestep exceed the allowed ratio
    difvar = settings.bsrn_vars['shortwave_diffuse']['name']
    dirvar = settings.bsrn_vars['shortwave_direct_horizontal_sac']['name']
    ghivar = settings.bsrn_vars['shortwave_global']['name']
    lwdvar = settings.bsrn_vars['longwave_down']['name']
    dim = settings.bsrn_vars['shortwave_diffuse']['dim']

    # calculate relative irradiances based on clear sky
    ghics = data.ghi_cs.interp(time_mcc=data[dim])
    ghics = xarray.where(ghics < 10, 10, ghics)

    # calculate relative ramp rates
    dif_rr = (data[difvar] / ghics).diff(dim=dim).__abs__() * 100
    dir_rr = (data[dirvar] / ghics).diff(dim=dim).__abs__() * 100
    ghi_rr = (data[ghivar] / ghics).diff(dim=dim).__abs__() * 100

    # prepare criteria thresholds
    dir_mean = data[dirvar].rolling({dim: 900}, center=True).mean()
    diffuse_limit = 5
    direct_limit = 20
    global_limit = xarray.where(dir_mean < 10, diffuse_limit, direct_limit)
    residual_limit_rel = 10  # % deviation from 100%
    residual_limit_abs = 20  # W/m2 absolute residual of ghi-(dir+dif)

    # detect invalid data based on thresholds
    invalid_dif = dif_rr > diffuse_limit
    invalid_dir = dir_rr > direct_limit
    invalid_ghi = ghi_rr > global_limit
    invalid_sw = ((invalid_dif | invalid_dir) | invalid_ghi).astype(np.uint)

    # reset false positives by checking correlation between direct and global
    dir_rr_m = dir_rr.rolling({dim: 30}, center=True).mean()
    ghi_rr_m = ghi_rr.rolling({dim: 30}, center=True).mean()
    rr_mean = (dir_rr_m - ghi_rr_m).__abs__()
    valid_sw = (rr_mean < 2.5)
    invalid_sw = xarray.where((invalid_sw == 1) & valid_sw, invalid_dif, invalid_sw)

    # pad the invalid measurement with 1.5 minutes before and after
    invalid_sw = invalid_sw.rolling(time_rad=180, center=True).mean()
    invalid_sw = xarray.where(invalid_sw > 0, 1, 0)
    invalid_sw = invalid_sw.reindex(time_rad=data[dim])

    # calculate the residual over 15 minute time frame
    res_rel = data[ghivar] / (data[dirvar] + data[difvar])
    res_rel = res_rel.rolling(time_rad=15 * 60, center=True, min_periods=5 * 60).mean().__abs__() * 100 - 100
    res_rel[res_rel > 100] = 100
    res_abs = data[ghivar] - (data[dirvar] + data[difvar])
    res_abs = res_abs.rolling(time_rad=15 * 60, center=True, min_periods=5 * 60).mean().__abs__()
    invalid_res = (res_rel > residual_limit_rel) & (res_abs > residual_limit_abs)
    invalid_sw = xarray.where(invalid_res, 1, invalid_sw)

    # and perhaps the simplest of all, require all three components to be there available
    invalid_na = data[ghivar].isnull() | data[difvar].isnull() | data[dirvar].isnull()
    invalid_sw = xarray.where(invalid_na, 1, invalid_sw)

    # compress to bytes
    invalid_sw = invalid_sw.astype(np.int8)

    # write filters to dataset
    data['custom_sw_quality'] = invalid_sw
    data['custom_sw_quality'].attrs['description'] = '0=good, 1=invalid dni/dif/ghi'
    data['custom_sw_quality'].attrs['long_name'] = 'short-wave data quality flags for 1 Hz (non-official)'
    # data['custom_lw_quality'] = invalid_lw
    # data['custom_lw_quality'].attrs['description'] = '0=good, 1=invalid lwd'

    if debug:
        data['dif_rr'] = dif_rr
        data['dir_rr'] = dir_rr
        data['ghi_rr'] = ghi_rr
        data['rr_mean'] = rr_mean
        data['res_rel'] = res_rel
        data['res_abs'] = res_abs
        data['global_limit'] = global_limit

    # export
    data.to_netcdf(fpath_tmp)
    os.rename(fpath_tmp, fpath)
    data.close()


@utils.timeit
def gap_filler(dt, res):
    """
    fill missing values/gaps in timeseries data

    :param dt:
    :param res:
    :return:
    """
    # generate filepaths
    fdir = settings.fdir_out.format(res=res, y=dt.year, m=dt.month)
    fname = dt.strftime('%Y%m%d.nc')
    fpath = os.path.join(fdir, fname)
    fpath_tmp = fpath.replace('.nc', '_tmp.nc')

    # load data
    if os.path.isfile(fpath):
        data = xarray.open_dataset(fpath)
    else:
        warnings.warn("Input file not found, skipping (%s)" % fpath)
        return

    # gap filling settings
    max_gap_size = timedelta(seconds=61)

    # fill the gaps
    for var in settings.bsrn_vars.values():
        var_ = var['name']
        if var_ in data:
            data[var_] = data[var_].interpolate_na(method='linear', use_coordinate=True, max_gap=max_gap_size,
                                                   dim=var['dim'])

    # export
    data.to_netcdf(fpath_tmp)
    os.rename(fpath_tmp, fpath)
    data.close()


def post_processing_cleanup(time_start, time_stop, update_version=False, rename_vars=False):
    """
    Remove deprecated variables from dataset that were introduced during debugging or previous versions

    :param time_start:
    :param time_stop:
    :param bool update_version: whether to update the dataset version to the latest
    :param rename_vars: rename variables from an older version (<8.1)
    :return:
    """
    # split the full requested timerange to process into separate days
    day_ranges = utils.split_dtrange_into_ranges(time_start, time_stop, ranges='days')

    vars_to_drop = ['D', 'F', 'res', 'tb', 'global_limit', 'res_abs', 'res_rel', 'dif_rr', 'dir_rr', 'ghi_rr',
                    'rr_mean']
    vars_to_rename = dict(dir='dni', dni_gmd='dhi_gmd', dni_sac='dhi_sac', solar_angle='solar_elev')
    update_counter = 0

    for i,  (ts, te) in enumerate(day_ranges):
        # generate filepaths
        fdir = settings.fdir_out.format(res='1sec', y=ts.year, m=ts.month)
        fname = ts.strftime('%Y%m%d.nc')
        fpath = os.path.join(fdir, fname)
        fpath_tmp = fpath.replace('.nc', '_tmp.nc')
        update = False

        # load data
        if os.path.isfile(fpath):
            data = xarray.open_dataset(fpath)
        else:
            warnings.warn("Input file not found, skipping (%s)" % fpath)
            continue

        # remove old variables
        for var in vars_to_drop:
            if var in data.variables:
                data = data.drop_vars([var])
                update = True

        # rename vars if requested
        if rename_vars:
            for var in vars_to_rename:
                if var in data.variables:
                    data = data.rename_vars({var: vars_to_rename[var]})
                    update = True

        # clean up attributes and variables
        for var_key in ['shortwave_global', 'shortwave_direct', 'shortwave_diffuse', 'longwave_down',
                        'relative_humidity', 'temperature_air', 'pressure']:
            var = settings.bsrn_vars[var_key]['name']
            if (var in data) and ('A' in data[var].attrs.keys()):
                data[var].attrs = {k: data[var].attrs[k] for k in ['units', 'long_name', 'Device'] if k in data[var].attrs}
                update = True

        # update version
        if update_version:
            if data.attrs['version'] != settings.VERSION:
                data.attrs['version'] = settings.VERSION
                update = True

        # export and close
        if update:
            data.to_netcdf(fpath_tmp)
            os.rename(fpath_tmp, fpath)
            update_counter += 1
        data.close()

        if i % 10 == 0:
            print("[%i/%i] Cleaning up preprocessed data: %s" % (i+1, len(day_ranges), ts))

    print('Done cleaning up preprocessed data, updated %i out of %i files' % (update_counter, len(day_ranges)))

