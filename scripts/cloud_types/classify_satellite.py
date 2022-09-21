import xarray
import os
from datetime import datetime, timedelta
import numpy as np

import general.settings as gsettings
import general.utils as gutils

# base settings
base_path = os.path.join(gsettings.fdir_research_data, 'eumetsat', 'msgcpp')
fdir_in_fmt = os.path.join(base_path, 'raw', '{y}', '{m:02d}')
fname_in_fmt = 'in%Y%m%d%H%M00305SVMSG<ID>.nc'
fdir_out_fmt = os.path.join(base_path, 'processed', 'fields', '{y}', '{m:02d}')
fname_out_fmt = 'class_%Y%m%d_%H%M.nc'

ids = {2014: '01UD', 2015: '01UD', 2016: 'E1UD'}


def load_data(dt):
    """
    load the data required by the classification algorithm
    :param dt:
    :return:
    """
    # load source data arrays
    fname_in = dt.strftime(fname_in_fmt).replace('<ID>', ids[dt.year])
    fdir_in = fdir_in_fmt.format(y=dt.year, m=dt.month)
    data_cot = xarray.open_dataset(os.path.join(fdir_in, 'CPP' + fname_in)).cot
    data_cth = xarray.open_dataset(os.path.join(fdir_in, 'CTX' + fname_in)).ctp

    # merge them into a dataset
    data = xarray.Dataset(data_vars={'ctp': data_cth, 'cot': data_cot})

    # add x and y coords so conditional selection can be done later
    if 'x' not in data.coords:
        data = data.assign_coords(x=data.x, y=data.y)
    return data


def classify_clouds(data):
    """
    Classify pixels into cloud type via 9-class classification based on cloud top pressure and optical thickness
    """
    # load and prepare data
    classes = xarray.DataArray(np.zeros(data.cot.shape), coords=dict(time=data.time, y=data.y, x=data.x))

    # classify
    classes = xarray.where((data.ctp < 440) & (data.cot < 3.6), 1, classes)
    classes = xarray.where((data.ctp < 440) & (data.cot >= 3.6), 2, classes)
    classes = xarray.where((data.ctp < 440) & (data.cot >= 23), 3, classes)

    classes = xarray.where((data.ctp >= 440) & (data.cot < 3.6), 4, classes)
    classes = xarray.where((data.ctp >= 440) & (data.cot >= 3.6), 5, classes)
    classes = xarray.where((data.ctp >= 440) & (data.cot >= 23), 6, classes)

    classes = xarray.where((data.ctp >= 680) & (data.cot < 3.6), 7, classes)
    classes = xarray.where((data.ctp >= 680) & (data.cot >= 3.6), 8, classes)
    classes = xarray.where((data.ctp >= 680) & (data.cot >= 23), 9, classes)

    # mask missing values and unclassified data
    classes = xarray.where(data.cot == 0, 0, classes)
    classes = xarray.where(data.cot.isnull(), -1, classes)

    # turn into a xarray with same format as the source data
    classes.name = 'cloud classes'
    classes.attrs['class_names'] = ['n/a', 'cloud-free', 'Ci', 'Cs', 'Cb', 'Ac', 'As', 'Ns', 'Cu', 'Sc', 'St']
    classes = classes.to_dataset(name='class')
    return classes


def classify_date(date, overwrite=False):
    # run the classification for a whole date
    dts = [date + timedelta(minutes=15 * i) for i in range(96)]
    classes = []

    for dt in dts:
        fpath_out = os.path.join(fdir_out_fmt.format(y=dt.year, m=dt.month), dt.strftime('classes_%Y%m%d_%H%M.nc'))
        if not os.path.isfile(fpath_out) or overwrite:
            try:
                data = load_data(dt)
                classes.append(classify_clouds(data))
            except FileNotFoundError:
                print('Source data not found, skipping: %s' % fpath_out)
                pass
            except OSError:
                print('Source data invalid, skipping: %s' % fpath_out)
        else:
            print('File exists, skipping: %s' % dt)

    # merge results and export
    classes = xarray.concat(classes, dim='time')
    fdir = fdir_out_fmt.format(y=date.year, m=date.month)
    os.makedirs(fdir) if not os.path.exists(fdir) else None
    classes.to_netcdf(os.path.join(fdir, date.strftime('classes_%Y%m%d.nc')))
    print('Processed classes for date: %s' % date)


if __name__ == "__main__":
    # classify_clouds(datetime(2018, 8, 11, 5, 30))
    # dts = gutils.generate_dt_range(datetime(2014, 1, 1), datetime(2017, 1, 1), delta_dt=timedelta(days=1))
    dts = [datetime(2014, 3, 20)]

    for dt in dts:
        classify_date(dt, overwrite=True)
