import xarray
from datetime import datetime, timedelta
import os
import numpy as np

from general import settings as gsettings
from general import utils as gutils

# settings of raw input files
fdir_msg_base = os.path.join(gsettings.fdir_research_data, 'eumetsat', 'msgcpp')
fdir_in_msg = os.path.join(fdir_msg_base, 'raw', '{y}', '{m:02d}')
fmt_in_msg = 'in%Y%m%d%H%M00305SVMSG<ID>.nc'
fdir_in_class = os.path.join(fdir_msg_base, 'processed', 'fields', '{y}', '{m:02d}')
fname_in_class = 'classes_%Y%m%d.nc'

ids = {2014: '01UD', 2015: '01UD', 2016: 'E1UD'}
msg_radii = [5, 10, 15]


def great_circle_distance(lon1, lat1, lon2, lat2):
    """
    calculate the absolute distance between two points on earth (assuming it's a sphere?)

    :param lon1: reference longitude (degrees)
    :param lat1: reference latitude (degrees)
    :param lon2: point longitude (degrees)
    :param lat2: point latitude (degrees)
    :return: distance to point (lon1, lat1) in kilometers
    """
    # convert to radians
    lon1, lat1, lon2, lat2 = map(np.deg2rad, [lon1, lat1, lon2, lat2])
    dlon = np.abs(lon2 - lon1)
    dlat = np.abs(lat2 - lat1)

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return 6367 * c * 1e3


def preprocess_claas_to_timeseries(year, target_latlon=(51.97, 4.92), method='constant', overwrite=False):
    """
    Turn raw CPP data from CLAAS2.1 dataset files into daily timeseries of single locations

    :param year: the year to process
    :param target_latlon:
    :param str method: retrieval method (only constant is supported)
    :param bool overwrite: whether to overwrite existing files
    :return:
    """
    # create the dates to process
    date = datetime(year, 1, 1)
    date_stop = datetime(year+3, 1, 1)
    prev_month = None
    radii = msg_radii
    time_radius = 10
    n_pixels = []

    while date < date_stop:
        # check whether data already exists
        fpath_out = gutils.generate_processed_fpath(date, which='msgcpp')
        if os.path.isfile(fpath_out) and not overwrite:
            date += timedelta(days=1)
            continue

        dts = [date + timedelta(minutes=15*i) for i in range(0, 24*4)]
        cot_max = [[] for _ in range(len(radii))]
        cot_mean = [[] for i in range(len(radii))]
        cot_std = [[] for i in range(len(radii))]
        cth_max = [[] for i in range(len(radii))]
        cth_mean = [[] for i in range(len(radii))]
        cth_std = [[] for i in range(len(radii))]

        # load coordinate file
        if prev_month != date.month:
            coord_fpath = os.path.join(fdir_in_msg.format(y=date.year, m=date.month), 'CM_SAF_CLAAS2_L2_AUX.nc')
            coords = xarray.open_dataset(coord_fpath, decode_times=False)

            # create distance field from coordinate file
            lat, lon = target_latlon
            lats, lons = coords.lat.values, coords.lon.values
            distance = great_circle_distance(lon1=lon, lat1=lat, lon2=lons, lat2=lats)

            # retrieve temporal offset
            scan_time_offset = int(xarray.where(distance < time_radius * 1e3, coords.tline, np.nan).mean())
            prev_month = date.month
            print('loaded coordinate file for month %s' % date)

        for dt in dts:
            # generate filepath
            fpath = os.path.join(fdir_in_msg.format(y=dt.year, m=dt.month), 'CPP' + dt.strftime(fmt_in_msg))
            fpath = fpath.replace('<ID>', ids[dt.year])

            # load cot data if possible
            if not os.path.isfile(fpath):
                dcot = None
            else:
                data = xarray.open_dataset(fpath)
                dcot = data.cot.squeeze(dim='time')

            # load cth data if possible
            fpath = fpath.replace('CPP', 'CTX')
            if not os.path.isfile(fpath):
                dcth = None
            else:
                data_ = xarray.open_dataset(fpath)
                dcth = data_.cth.squeeze(dim='time')

            # process data, if possible, for each radius
            for i, radius in enumerate(radii):
                if dcot is None and dcth is None:
                    cot_max[i].append(np.nan)
                    cot_mean[i].append(np.nan)
                    cot_std[i].append(np.nan)
                    cth_max[i].append(np.nan)
                    cth_mean[i].append(np.nan)
                    cth_std[i].append(np.nan)
                else:
                    # create the correct subset
                    cot = xarray.where(distance < radius * 1e3, dcot, np.nan)
                    cth = xarray.where(distance < radius * 1e3, dcth, np.nan)

                    # add to pixel count array for the first round
                    if len(n_pixels) < len(radii):
                        n_pixels.append((distance < radius * 1e3).sum())

                    # create a test set to check if nans exist within radius
                    cot_test = xarray.where(distance < radius * 1e3, cot, -1)
                    cth_test = xarray.where(distance < radius * 1e3, cth, -1)
                    nan_count_a = cot_test.isnull().sum() / n_pixels[i] if dcot is not None else 1
                    nan_count_b = cth_test.isnull().sum() / n_pixels[i] if dcth is not None else 1

                    # retrieve cot and cth statistics for subset
                    if nan_count_b > 0.75 or nan_count_a > 0.75:
                        cot_max[i].append(np.nan)
                        cot_mean[i].append(np.nan)
                        cot_std[i].append(np.nan)
                        cth_max[i].append(np.nan)
                        cth_mean[i].append(np.nan)
                        cth_std[i].append(np.nan)
                    else:
                        cot_max[i].append(float(cot.max(dim=('x', 'y'))))
                        cot_mean[i].append(float(cot.mean(dim=('x', 'y'))))
                        cot_std[i].append(float(cot.std(dim=('x', 'y'))) if cot_mean[-1] != 0 else np.nan)
                        cth_max[i].append(float(cth.max(dim=('x', 'y'))))
                        cth_mean[i].append(float(cth.mean(dim=('x', 'y'))))
                        cth_std[i].append(float(cth.std(dim=('x', 'y'))) if cth_mean[-1] != 0 else np.nan)

        # process the cloud classes
        dclasses = xarray.open_dataset(os.path.join(fdir_in_class.format(y=date.year, m=date.month),
                                                    date.strftime(fname_in_class)))['class']
        c_attrs = dclasses.attrs
        ccover = []
        dom_class = []

        for i, radius in enumerate(radii):
            # prepare classes
            classes = xarray.where(distance < radius * 1e3, dclasses, -1)
            if len(classes.time) != len(dts):
                classes = classes.reindex(time=dts)

            # calculate class cover (aka cloud cover)
            x, y = classes.shape[1:]
            ccover.append((classes > 0).sum(dim=('x', 'y')) / (x*y - (classes == -1).sum(dim=('x', 'y'))))

            # find dominant class within subset
            cat_weight = []
            classes = xarray.where(distance >= radius * 1e3, -2, classes)
            classes_nrs = range(1, 10)
            for cat in classes_nrs:
                cat_weight.append((classes == cat).sum(dim=('x', 'y')))
            no_class = (classes == 0).sum(dim=('x', 'y')) == n_pixels[i]
            no_data = (classes == -1).sum(dim=('x', 'y')) == n_pixels[i]
            dom_class_ = np.array([classes_nrs[int(i)] for i in np.argmax(np.array(cat_weight), axis=0)])
            dom_class_[no_class] = 0
            dom_class_[no_data] = -1

            # dom_class_[np.isnan(cot_mean[i])] = 0
            dom_class.append(dom_class_)

        # create new dts axis with correct offsets based on scan time
        dts = [dt + timedelta(seconds=scan_time_offset) for dt in dts]

        # create the output dataset
        coords = dict(radius=radii, datetime=dts)
        cot_max = xarray.DataArray(data=cot_max, coords=coords)
        cot_mean = xarray.DataArray(data=cot_mean, coords=coords)
        cot_std = xarray.DataArray(data=cot_std, coords=coords)
        cth_max = xarray.DataArray(data=cth_max, coords=coords)
        cth_mean = xarray.DataArray(data=cth_mean, coords=coords)
        cth_std = xarray.DataArray(data=cth_std, coords=coords)
        ccover = xarray.DataArray(data=ccover, coords=coords)
        dom_class = xarray.DataArray(data=dom_class, coords=coords)
        n_pixels = xarray.DataArray(data=n_pixels, coords=dict(radius=radii))

        data_out = xarray.Dataset(data_vars=dict(cot_max=cot_max, cot=cot_mean, cot_std=cot_std,
                                                 cth_max=cth_max, cth=cth_mean, cth_std=cth_std,
                                                 ccover=ccover, dom_class=dom_class, n_pixels=n_pixels))
        # add attributes
        data_out.attrs = data.attrs
        data_out.radius.attrs['long_name'] = 'selection radius'
        data_out.radius.attrs['units'] = 'km'
        data_out.radius.attrs['method'] = method
        data_out.n_pixels.attrs['long_name'] = 'number of pixels within radius selectin'
        data_out.cot.attrs['long_name'] = 'cloud optical thickness mean'
        data_out.cot_std.attrs['long_name'] = 'cloud optical thickness standard deviation'
        data_out.cot_max.attrs['log_name'] = 'cloud optical thickness max'
        data_out.cth.attrs['long_name'] = 'cloud top height mean'
        data_out.cth_std.attrs['long_name'] = 'cloud top height standard deviation'
        data_out.cth_max.attrs['long_name'] = 'cloud top height max'
        data_out.dom_class.attrs['long_name'] = 'dominant cloud classification (ISCCP)'
        data_out.dom_class.attrs = c_attrs
        data_out.ccover.attrs['long_name'] = 'cloud cover (fraction of pixels with clouds)'

        # export
        os.makedirs(os.path.dirname(fpath_out)) if not os.path.exists(os.path.dirname(fpath_out)) else None
        data_out.to_netcdf(fpath_out)

        print('Processed %s' % fpath_out)
        date += timedelta(days=1)


def preprocess_nubiscope(year=2016, overwrite=True):
    """
    Preprocess monthly nubiscope data into ready-to-use daily files

    :param year:
    :param overwrite:
    :return:
    """
    # create the dates to process
    date = datetime(year, 1, 1)
    date_stop = datetime(year+1, 1, 1)
    prev_month = None

    while date < date_stop:
        fpath_out = gutils.generate_processed_fpath(date, which='nubiscope', raw=False)

        if os.path.isfile(fpath_out) and not overwrite:
            print("File exists, not overwriting: %s" % fpath_out)
            date += timedelta(days=1)
            continue

        if date.month != prev_month:
            # load month
            fpath = gutils.generate_processed_fpath(date, which='nubiscope', raw=True)
            data = xarray.open_dataset(fpath)
            prev_month = date.month

            # preprocess the month
            overcast = xarray.where(data.obscuration_type == b'OC', 1, 0)
            overcast = xarray.where(data.obscuration_type == b'HP', 1, overcast)
            overcast = xarray.where(data.obscuration_type == b'DF', 1, overcast)
            overcast = xarray.where(data.cldcover_total.isnull(), np.nan, overcast)

            clearsky = xarray.where(data.obscuration_type == b'CS', 1, 0)
            clearsky = xarray.where(data.cldcover_total.isnull(), np.nan, clearsky)

            data['overcast'] = overcast
            data['overcast'].attrs['long_name'] = 'Derived overcast class (1=overcast+highprecip+densefog, 0=other)'

            data['clearsky'] = clearsky
            data['clearsky'].attrs['long_name'] = 'Derived clearsky class (1=clearsky, 0=other)'

        # cut up into daily files
        data_subset = data.sel(time=slice(date, date+timedelta(days=1)))

        if len(data_subset.time) > 0:
            data_subset = data_subset.rename({'time': 'datetime'})

            # create output fpath and export
            if not os.path.exists(os.path.dirname(fpath_out)):
                os.makedirs(os.path.dirname(fpath_out))
            data_subset.to_netcdf(fpath_out)

            print('Processed nubiscope for %s' % date)
        else:
            print("No datapoints for date: %s, skipping" % date)
        date += timedelta(days=1)


if __name__ == "__main__":
    preprocess_claas_to_timeseries(year=2014, overwrite=True)
    # preprocess_nubiscope(year=2015, overwrite=False)
