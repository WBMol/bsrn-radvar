import xarray
from general import utils as gutils
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import os

import settings


def get_u200_above_angle(date, angle):
    """
    Calculate the mean 200 meter wind speed while solar elevation angle is above a certain value for a given date

    :param date: datetime, used to load preprocessed radiation dataset
    :param angle: solar elevation angle in degrees
    :return:
    """
    # load time series data
    fpath = gutils.generate_processed_fpath(date, '1sec')
    data = xarray.open_dataset(fpath)[['solar_angle', 'wspd']]

    # find the datetime range over which the solar elevation angle criterium is satisfied
    valid_data = xarray.where(data.solar_angle > angle, data.solar_angle, np.nan).dropna(dim='time_rad')
    dtime_range = slice(*valid_data.time_rad.isel(time_rad=[0, -1]))

    # slice the 200 meter wind speed to this range
    u200 = data.wspd.sel(time_tower=dtime_range, z=200)

    # get daylight duration
    duration = int((dtime_range.stop - dtime_range.start).astype(np.uint) / 1e9)

    # return the mean for this day
    return duration, float(u200.mean(dim='time_tower'))


def length_for_doy(angle=5):
    """
    Calculate for a range of dates and a minimum solar elevation angle the daylength duration and mean wind speed
    Hardcoded to 2020

    :param angle:
    :return:
    """
    dts = gutils.generate_dt_range(datetime(2012, 1, 1), datetime(2021, 1, 1))
    ls = []
    u200 = []
    doy = []

    for i, dt in enumerate(dts):
        try:
            dur, daywind = get_u200_above_angle(dt, angle=angle)
        except FileNotFoundError:
            dur, daywind = np.nan, np.nan
        ls.append(dur)
        u200.append(daywind)
        doy.append(int((dt - dts[0]).total_seconds() / 86400))
        if i % 10 == 0 or i == len(dts):
            print('Processed %s' % dt.strftime('%Y-%m-%d'))

    with open(os.path.join(settings.fdir_results, 'day_length_lut.txt'), 'w') as f:
        f.write('date,doy,length,u200\n')
        for x in zip(dts, doy, ls, u200):
            try:
                f.write('%s,%i,%i,%.2f\n' % x)
            except ValueError:
                f.write('%s,%i,nan,nan\n' % (x[0], x[1]))


def fit_func():
    # load obs
    lut = pd.read_csv(os.path.join(settings.fdir_results, 'day_length_lut.txt'), delimiter=',')
    plt.plot(lut.doy, lut.length)

    # fit model
    y = np.cos((lut.doy + 11 + 365/2) / 365 * (2 * np.pi)) * 17.5e3 + 37.5e3

    plt.plot(lut.doy, y)


def calculate_daylength_size():
    """
    Calculate the length of days in meters throughout the year based on 200 meter wind speed, used for power law fitting

    :return:
    """
    # load day length duration for 2020
    lut = pd.read_csv(os.path.join(settings.fdir_results, 'day_length_lut.txt'), delimiter=',', index_col=0,
                      parse_dates=True)

    # create subsets
    lut_summer = lut[((lut.index.month >= 5) & (lut.index.month < 9))]
    lut_winter = lut[((lut.index.month >= 11) | (lut.index.month < 3))]

    # export results
    os.makedirs(settings.fdir_results) if not os.path.exists(settings.fdir_results) else None
    with open(os.path.join(settings.fdir_results, 'day_length_climatology.txt'), 'w') as f:

        for data, label in zip([lut, lut_summer, lut_winter], ['all months ', 'summer (5-8)', 'winter (11-02)']):
            f.write('%s\t Mean duration: %i s, mean u200: %.2f m/s, mean size: %.2f km\n' % (
                label, data.length.mean(), data.u200.mean(), (data.length * data.u200).mean() / 1e3
            ))


if __name__ == "__main__":
    # length_for_doy(angle=5)
    # fit_func()
    calculate_daylength_size()
