# Event Statistics Processor
Calculating all sorts of statistics for daily radiation time series that are classified. 
Note that 'segment' is just a technical term used, and is equivalent to 'event'.

Script settings are in `settings.py`, after which you are ready to run the main script:
- `segment_statistics.py` 

This generates two files for a requested classes (e.g. 'shadow'):
1. `1sec_segments_daily_stats_all_<class>.nc` = aggregated statistics at daily level (total count and duration)
2. `1sec_segment_total_stats_all_<class>.nc` = all events of a class with each event's stats
