from datetime import datetime

cases = {
    'unphysical': [
        datetime(2011, 6, 17),
        datetime(2013, 1, 2),
        datetime(2013, 1, 16),  # snow and cirrus, questionable
        datetime(2014, 5, 14),  # large residual sum
        datetime(2016, 4, 11),
        datetime(2017, 1, 2),
        datetime(2017, 1, 4),
        datetime(2017, 1, 6),
        datetime(2018, 1, 3),
        datetime(2018, 4, 16),
        datetime(2018, 7, 2),
        datetime(2018, 7, 4),
        datetime(2018, 5, 25),
        datetime(2018, 8, 3),
    ],
    'physical': [
        datetime(2011, 6, 26),
        datetime(2011, 6, 29),
        datetime(2015, 7, 27),
        datetime(2017, 1, 3),
        datetime(2018, 1, 1),
        datetime(2018, 1, 18),
    ],
    'mcclear': [
        datetime(2019, 1, 30)
    ],
}
