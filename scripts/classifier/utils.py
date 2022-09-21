from datetime import datetime

test_cases = {
    'clearsky': [
        datetime(2016, 1, 8),
        datetime(2016, 1, 17),
        datetime(2016, 1, 25),
        datetime(2016, 5, 12),
        datetime(2016, 6, 6),
        datetime(2016, 6, 9),
        datetime(2016, 6, 16),
        datetime(2019, 7, 6)
    ],
    'cs_to_oc': [
        datetime(2014, 5, 13),
        datetime(2015, 3, 22),
        datetime(2015, 4, 23),
        datetime(2015, 7, 20),
        datetime(2016, 9, 25)
    ],
    'paper_showcase': [
        datetime(2015, 4, 23)
    ]
}


def get_time_plot_range(sea, pad=0, res=0.25, sea_limit=0):
    """
    determine the min and max hours of day to plot based on the solar elevation angle

    :param xarray.DataArray sea:  solar elevation angles
    :param int pad: amount of iterations to pad (in units of resolution)
    :param float res: resolution of  rounding in hours
    :param int sea_limit: minimum value of solar elevation angle
    :return:
    """
    # select the hour and minutes of lower and upper end in the valid solar elevation angle range
    sea = sea.where(sea > sea_limit, drop=True)
    hr_min = sea.time_rad.dt.hour[0] + sea.time_rad.dt.minute[0] / 60
    hr_max = sea.time_rad.dt.hour[-1] + sea.time_rad.dt.minute[-1] / 60

    # round them up or down
    hr_min -= hr_min % res
    hr_max += (res - hr_max % res)

    # add some padding to either side and return
    return float(hr_min) - pad*res, float(hr_max) + pad*res
