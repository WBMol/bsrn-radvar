# Radiation Time Series Classification

Main processing steps
1. `settings.py` for the input location of irradiance data
2. `classify.py` for running the classification algorithm on a single date or range of dates
3. `visual_validation.py` for plotting the classification in a time series overview per date, used to visually validate

Extras (applicable to weather-classes only)
4. `preprocess_validation.py` for generating a validation dataset based on MSGCPP and Nubiscope
5. `statistical_validation.py` calculate skill scores based on validation dataset