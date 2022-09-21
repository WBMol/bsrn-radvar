import xarray
import os
import numpy as np
from datetime import datetime, timedelta
from glob import glob
import warnings

import scripts.classifier.settings as settings
import scripts.classifier.utils as utils
from utils import test_cases

import general.utils as gutils


class MeasurementClassifier:

    def __init__(self, data, date=None):
        self.data = data
        self.date = date
        self.ghi_cs = self.data.ghi_cs.interp(time_mcc=data.time_rad)
        self.dif_cs = self.data.dif_cs.interp(time_mcc=data.time_rad)

    def get_dummy_data_array(self):
        return xarray.DataArray(np.zeros(self.data.time_rad.shape), [('time_class', self.data.time_rad.values)])

    def sunshine(self):
        """
        The WMO definition for sunshine is when the direct irradiance exceeds 120 W/m2
        :return:
        """
        return (self.data.dni >= 120).values

    def cloud_enhancement(self):
        """
        New filter that select any radiation event above clear-sky in addition to non-zero direct sunlight.
        Stricter criteria can be selected at a later stage when analysing the events.

        No rolling window; CE effects are instantaneous
        :return:
        """
        # find any data point that exceeds clear-sky
        c1 = (self.data.ghi / self.ghi_cs > 1.001).values
        c2 = (self.data.dhi_sac > 10).values
        ce = np.logical_and(c1, c2)

        # set the activation threshold (1%)
        c3 = (self.data.ghi / self.ghi_cs > 1.01).values
        c4 = (self.data.ghi - self.ghi_cs > 10).values
        ce_act = np.logical_and(c3, c4)

        # expand the ce_act segments to the boundaries of ce_all
        diffs = np.diff(ce)
        changes = np.where(np.abs(diffs) > 0)[0] + 1

        for i, seg in enumerate(changes):
            j, k = changes[i], changes[i+1 if i < len(changes)-1 else -1]
            if np.sum(ce_act[j:k]) == 0:
                ce[j:k] = False

        return ce

    def solar_angle_filter(self, maximum_angle=0):
        """
        returns an array indicating which solar elevation angles are below the maximum_angle (True)

        :param float maximum_angle: maximum solar elevation angle to be considered True
        :return:
        """
        return (self.data.solar_elev < maximum_angle).values

    def clear_sky_filter(self, return_intermediate_values=False):
        """
        New filter to identify clear-sky

        :param bool return_intermediate_values: not implemented yet.
        :return:
        """
        minutes = 15  # base window size for the clear sky filter

        if self.data.resolution == '1min':
            thresholds = {
                'crit_1': 0.01,
                'crit_2': 0.01,
            }
            periods = minutes
        else:
            thresholds = {
                'crit_1': 0.03,
                'crit_2': 0.01,
                'crit_3': 5
            }
            periods = minutes * 60

        periods_min = int(0.25 * periods)  # 25% data required at a minimum for valid results
        center = True

        cmf = self.data.ghi / self.ghi_cs
        cmf = xarray.where(self.ghi_cs > 10, cmf, np.nan)
        cmf_rolling = cmf.rolling(time_rad=periods, center=center, min_periods=periods_min)
        res = (self.data.ghi - self.ghi_cs).__abs__()
        res_rolling = res.rolling(time_rad=periods, center=center, min_periods=periods_min)

        # calculate arrays used for criteria
        arr_1 = (cmf_rolling.max() - 1).__abs__()
        arr_1 = xarray.where(arr_1 < 0.05, arr_1, 0.05)  # limit the residuals to 5%
        arr_2 = cmf_rolling.std()
        arr_2 = xarray.where(arr_2 < 0.05, arr_2, 0.05)  # same here, limit stds to 5%
        arr_3 = res_rolling.max()
        arr_3 = xarray.where(arr_3 < 20, arr_3, 20)  # limit to 20 w/m2 deviations

        # within x% of clear-sky and certain smoothness
        crit_1 = arr_1 < thresholds['crit_1']
        crit_2 = arr_2 < thresholds['crit_2']
        crit_3 = arr_3 < thresholds['crit_3']

        if return_intermediate_values:
            # add long name attributes to data arrays
            arr_1.attrs['long_name'] = 'max relative difference'
            arr_2.attrs['long_name'] = 'std relative different'
            arr_3.attrs['long_name'] = 'max absolute different'

            # combine all relevant data into a dataset
            data = xarray.Dataset(data_vars={
                'crit_1': crit_1, 'crit_1_var': arr_1,
                'crit_2': crit_2, 'crit_2_var': arr_2,
                'crit_3': crit_3, 'crit_3_var': arr_3,
            })

            # add threshold attributes
            for i in range(1, 4):
                cvar = 'crit_%i' % i
                data[cvar].attrs['threshold'] = thresholds[cvar]

            # export
            fdir = os.path.join(settings.fdir_data_out_debug, self.data.resolution)
            os.makedirs(fdir) if not os.path.exists(fdir) else None
            fpath = os.path.join(fdir, self.date.strftime('%Y%m%d_clearsky.nc'))
            data.to_netcdf(fpath)

        is_clearsky = (crit_1 | crit_3) & crit_2  # if within % OR abs value, AND not variable, then yes.
        return is_clearsky.values

    def overcast_filter(self, return_intermediate_values=False):
        """
        filter for identifying overcast conditions, where it is overcast of barely any direct sunlight hits within
        a 60 minute window
        :return:
        """
        minutes = 45

        if self.data.resolution == '1min':
            raise NotImplementedError("1 min not implemented for overcast filter")
        else:
            thresholds = {
                'crit_1': 0.01,
                'crit_2': 10,
            }
            periods = minutes * 60

        periods_min = int(0.25 * periods)
        center = True

        # suppress RuntimeWarning of mean of empty slice: some rolling windows are filled with nans, which is expected
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            # do some quality control over the direct irradiance
            dir_cleaned = self.data['dni']
            dir_cleaned[dir_cleaned <= 2] = 0.
            sum_dir = dir_cleaned.rolling(time_rad=periods, center=center, min_periods=periods_min).sum()
            sum_cs = self.ghi_cs.rolling(time_rad=periods, center=center, min_periods=periods_min).sum()

        # first criteria, whether there's any direct at all
        dir_ratio = (sum_dir / sum_cs)
        dir_ratio[sum_cs < 1] = 0.1
        dir_ratio[dir_ratio > 0.1] = 0.1
        crit_1 = dir_ratio < thresholds['crit_1']

        # second criteria, absolute difference between ghi_cs and dif for edge cases
        crit_2 = sum_dir / 3600 < thresholds['crit_2']

        # combine criteria
        is_overcast = crit_1 & crit_2

        if return_intermediate_values:
            # add names to the criteria data array
            dir_ratio.attrs['long_name'] = 'direct fraction of clear-sky sky'
            sum_dir.attrs['long_name'] = 'direct irradiance mean'

            # add the data arrays to a dataset
            data = xarray.Dataset(data_vars={
                'crit_1': crit_1, 'crit_1_var': dir_ratio,
                'crit_2': crit_2, 'crit_2_var': sum_dir / 3600,
            })

            # add threshold attributes
            for i in range(1, 3):
                cvar = 'crit_%i' % i
                data[cvar].attrs['threshold'] = thresholds[cvar]

            # export
            fdir = os.path.join(settings.fdir_data_out_debug, self.data.resolution)
            os.makedirs(fdir) if not os.path.exists(fdir) else None
            fpath = os.path.join(fdir, self.date.strftime('%Y%m%d_overcast.nc'))
            data.to_netcdf(fpath)

        return is_overcast.values

    def cumulus_filter(self, return_intermediate_values=False):
        """

        :return:
        """
        minutes = 60

        if self.data.resolution == '1min':
            periods = minutes
            thresholds = {
                'crit_1': 3,
                'crit_2': 0.30,
                'crit_3': 0.15,
                'crit_4': 7
            }
        else:
            thresholds = {
                'crit_1': 3,
                'crit_2': 0.25,
                'crit_3': 0.15,
                'crit_4': 15
            }
            periods = minutes * 60

        periods_min = int(0.25 * periods)
        center = True

        # set up some variables
        t_target = self.data.time_rad
        ghi_cs = self.data.ghi_cs.interp(time_mcc=t_target).drop_vars('time_mcc')
        ghi_cs[np.isnan(self.data.ghi)] = np.nan
        # delta_t = float((t_target[1] - t_target[0]).values) / 60e9

        # suppress RuntimeWarning of mean of empty slice: some rolling windows are filled with nans, which is expected
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)

            ghi_norm = self.data.ghi / ghi_cs
            ghi_norm = ghi_norm.where(ghi_cs > 50)

            # custom normalized variability index
            vi = ((ghi_norm*100).diff(dim='time_rad') ** 2).reindex(time_rad=t_target)
            vi = vi.rolling(time_rad=periods, center=center, min_periods=periods_min).mean()

            # time spent 5% above clear sky
            ce = (ghi_norm > 1.03).astype(np.uint)  # rolling can't deal with casting bools to ints
            ce = ce.rolling(time_rad=periods, center=center, min_periods=periods_min).mean()

            # time spent as shadow
            sh = (self.data.dir < 120).astype(np.uint)
            sh = sh.rolling(time_rad=periods, center=center, min_periods=periods_min).mean()

            # sign changes
            above, below = ghi_norm >= 1, ghi_norm < 1
            sc = ghi_norm
            sc[above] = 1
            sc[below] = 0
            sc = (sc.diff(dim='time_rad').__abs__()).reindex(time_rad=t_target)
            sc = sc.rolling(time_rad=periods, center=center, min_periods=periods_min).sum()

            # variability index
            # vi_num = (self.data.ghi.diff(dim='time_rad') ** 2 + 1 ** 2) ** 0.5
            # vi_den = (ghi_cs.diff(dim='time_rad') ** 2 + 1 ** 2) ** 0.5
            # vi_num = vi_num.rolling(time_rad=periods, center=center, min_periods=periods_min).sum()
            # vi_den = vi_den.rolling(time_rad=periods, center=center, min_periods=periods_min).sum()
            # vi_den = vi_den.where(vi_den > 5)

            # cloud modification factor
            # cmf_num = self.data.ghi.rolling(time_rad=periods, center=center, min_periods=periods_min).sum()
            # cmf_den = ghi_cs.rolling(time_rad=periods, center=center, min_periods=periods_min).sum()

        # first criteria, whether there's any direct at all
        # vi = (vi_num / vi_den).reindex(time_rad=t_target)
        # vi = vi.where((vi > 0))
        crit_1 = vi > thresholds['crit_1']

        # second criteria, absolute difference between ghi_cs and dif for edge cases
        # cmf = cmf_num / cmf_den
        # cmf = cmf.where((cmf > 0) & (cmf < 1.1))
        crit_2 = ce > thresholds['crit_2']

        # third criterion
        crit_3 = sh > thresholds['crit_3']

        # fourth
        crit_4 = sc > thresholds['crit_4']

        # combine criteria
        is_cumulus = crit_1 & crit_2 & crit_3 & crit_4

        if return_intermediate_values:
            # add names to the criteria data array
            vi.attrs['long_name'] = 'normalized variability'
            ce.attrs['long_name'] = 'time spent above 105% clear-sky'
            sh.attrs['long_name'] = 'time spent below 120 Wm-2 dir'
            sc.attrs['long_name'] = 'amount of sign changes'

            # add the data arrays to a dataset
            data = xarray.Dataset(data_vars={
                'crit_1': crit_1, 'crit_1_var': vi,
                'crit_2': crit_2, 'crit_2_var': ce,
                'crit_3': crit_3, 'crit_3_var': sh,
                'crit_4': crit_4, 'crit_4_var': sc
            })

            # add threshold attributes
            for i in range(1, 5):
                cvar = 'crit_%i' % i
                data[cvar].attrs['threshold'] = thresholds[cvar]

            # export
            fdir = os.path.join(settings.fdir_data_out_debug, self.data.resolution)
            os.makedirs(fdir) if not os.path.exists(fdir) else None
            fpath = os.path.join(fdir, self.date.strftime('%Y%m%d_cumulus.nc'))
            data.to_netcdf(fpath)

        return is_cumulus.values

    def variable_filter(self, ce, shadow, return_intermediate_values=False):
        """
        Variable weather filter, which counts the amount of CE and Shadows within a time frame

        :param ce:
        :param shadow:
        :param return_intermediate_values:
        :return:
        """
        # create a ce-shadow-only array
        cs = xarray.where(ce, 1, np.nan)
        cs = xarray.where(shadow, 0, cs)
        cs = xarray.DataArray(cs, coords=dict(time=range(len(cs))))
        cs = cs.interpolate_na(dim='time', method='nearest')

        # detect transitions
        cst = cs.diff(dim='time').__abs__().reindex(time=cs.time)

        # count the amount of transitions per hour
        cst_per_hour = cst.rolling(time=3600, min_periods=1800, center=True).sum()

        if return_intermediate_values:
            fdir = os.path.join(settings.fdir_data_out_debug, self.data.resolution)
            os.makedirs(fdir) if not os.path.exists(fdir) else None
            fpath = os.path.join(fdir, self.date.strftime('%Y%m%d_variable.nc'))

            ds = xarray.Dataset(data_vars=dict(cst_per_hour=cst_per_hour, cst=cst, cs=cs))
            ds.to_netcdf(fpath)

        # return the classification
        return cst_per_hour.values > 10

    def shadow_filter(self):
        """
        shadow class based on WMO definition of sunlight: anything below 120 W/m2 of DIR gets marked as shadow
        :return:
        """
        return (self.data.dni < 120).values

    def missing_data_filter(self):
        """
        return a boolean dataarray to mark measurements that have missing values in any of the 3 measurement components

        :return:
        """
        return (self.data.ghi.isnull() | self.ghi_cs.isnull() | self.data.dni.isnull()).values


def run_classification(date, debug=False, res='1sec'):
    """
    Run the classification algorithm for a single date.

    :param datetime.datetime date: datetime to process (only y, m, d is used)
    :param bool debug: whether debug mode is on, this will export intermediate values of classifications
    :param str res: which resolution dataset to run with (1min or 1sec)
    :return: success
    :rtype: bool
    """
    # load data
    fpath_in = gutils.generate_processed_fpath(date, res=res, which='bsrn')
    if not os.path.isfile(fpath_in):
        return False
    data = gutils.load_timeseries_data(fpath_in)

    if float(data.attrs['version']) < float(settings.MIN_VERSION_REQ):
        warnings.warn("Data version (%s) is below the minimum required version (%s), skipping." %
                      (data.attrs['version'], settings.MIN_VERSION_REQ))
        return False

    # initialize the classifier with measurement data
    classifier = MeasurementClassifier(data=data, date=date)

    # retrieve two criteria to be marked as n/a
    is_night = classifier.solar_angle_filter(maximum_angle=0)
    is_missing = classifier.missing_data_filter()

    # Group two class filters
    is_sunny_wmo = classifier.sunshine()
    is_ce = classifier.cloud_enhancement()
    is_shadow = classifier.shadow_filter()

    # combine to one clear sky
    group_2_classes = classifier.get_dummy_data_array()
    group_2_classes[is_shadow] = 2
    group_2_classes[is_sunny_wmo] = 3
    group_2_classes[is_ce] = 4
    group_2_classes[is_night | is_missing] = 1
    group_2_classes.attrs['class_names'] = ['no class', 'no data', 'shadow', 'sunshine', 'cloud-enh.']

    # Group one class filters
    is_clearsky = classifier.clear_sky_filter(return_intermediate_values=debug)
    is_overcast = classifier.overcast_filter(return_intermediate_values=debug)
    is_cumulus = classifier.variable_filter(ce=is_ce, shadow=is_shadow, return_intermediate_values=debug)

    group_1_classes = classifier.get_dummy_data_array()
    group_1_classes[is_clearsky] = 2
    group_1_classes[is_overcast] = 3
    group_1_classes[is_cumulus] = 4
    group_1_classes[is_night | is_missing] = 1
    group_1_classes.attrs['class_names'] = ['no class', 'no data', 'clearsky', 'overcast', 'variable']

    # Combine into original dataset dataset
    data.close()
    data = gutils.load_timeseries_data(fpath_in, apply_quality_control=False)
    data['classes_group_1'] = group_1_classes.astype(np.int8)
    data['classes_group_2'] = group_2_classes.astype(np.int8)

    # export the data
    fpath_out_tmp = fpath_in.replace('.nc', '.tmp.nc')
    data.to_netcdf(fpath_out_tmp)
    os.rename(fpath_out_tmp, fpath_in)
    return True


def batch_run_range(dt_start=None, dt_stop=None, dts=None, debug=False, res='1min'):
    """
    run a bunch of dates at once
    :param dt_start: start of date range
    :param dt_stop: end of date range
    :param list dts: list of dates
    :param bool debug: classify in debug mode (outputs intermediate classification data)
    :param str res: resolution dataset (1min or 1sec)
    :return:
    """
    if dts is None:
        dts = gutils.generate_dt_range(dt_start, dt_stop, delta_dt=timedelta(days=1))

    for i, dt in enumerate(dts):
        success = run_classification(date=dt, debug=debug, res=res)
        if success:
            print('Processed %s/%s, date: %s' % (i + 1, len(dts), dt))
        else:
            print('Skipped %s' % dt)


def batch_run_all_available(res='1min', debug=False):
    """
    run all available bsrn

    :param str res: 1min or 1sec as temporal resolution of BSRN Cabauw data
    :param bool debug: classify in debug mode (outputs intermediate classification data)
    :return:
    """
    files = []
    for year in range(2015, 2021):
        for month in range(1, 13):
            fdir = settings.fdir_data_in_fmt.format(res=res, y=year, m=month)
            files += sorted(glob(os.path.join(fdir, '*.nc')))
    dts = [datetime.strptime(fpath.split('/')[-1], '%Y%m%d.nc') for fpath in files]

    for i, dt in enumerate(dts):
        success = run_classification(date=dt, debug=debug, res=res)
        if success:
            print('Processed %s/%s, date: %s' % (i + 1, len(dts), dt.strftime('%Y-%m-%d')))
        else:
            print('Failed to process date: %s' % (dt.strftime('%Y-%m-%d')))


if __name__ == "__main__":
    # datetime to process
    # dt = datetime(2019, 7, 6)

    # process data
    # run_classification(date=dt, debug=True, res='1sec')

    # batch_run_range(datetime(2011, 1, 1), datetime(2020, 1, 1), res='1sec')
    batch_run_range(datetime(2018, 8, 1), datetime(2018, 8, 6), res='1sec')
    # batch_run_range(dts=test_cases['clearsky'], res='1sec')

    # batch_run_all_available()
