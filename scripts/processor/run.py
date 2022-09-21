from datetime import datetime

import scripts.processor.preprocess as preproc


if __name__ == "__main__":
    # period = 'cleanup'
    period = 'year'

    if period == 'year':
        time_start = datetime(2021, 1, 1)
        time_end = datetime(2022, 1, 1)
        preproc.preprocess_data(time_start, time_end)
    elif period == 'qc_test':
        preproc.specific_quality_check_routine(subset='test')
    elif period == 'qc_all':
        preproc.specific_quality_check_routine(subset=(datetime(2011, 1, 1), datetime(2020, 1, 1)))
    elif period == 'cleanup':
        ts = datetime(2011, 2, 1)
        te = datetime(2021, 1, 1)
        preproc.post_processing_cleanup(ts, te, update_version=True)
