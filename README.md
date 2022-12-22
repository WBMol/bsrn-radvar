# Radiation Time Series Processing

This repository is a collection of tools to process 1 Hz BSRN solar irradiance time series, classification and analyses.
This is the codebase used to produce the datasets published at Zenodo:
- [Part 1: pure measurements](https://doi.org/10.5281/zenodo.7093163)
- [Part 2: complementary data, statistics, classification, etc](https://doi.org/10.5281/zenodo.7092057)

## Structure

Scripts and tools are grouped in directories, each directory should have a readme briefly explaining its contents.
The structure is as follows:

| Directory     | Contains                                                      |
|---------------|---------------------------------------------------------------|
| paper-figures | directory with scripts that generate figures for publications |
| general       | contains project-wide utilities and settings                  |
| scripts       | a collection of processing and analysis scripts               |

In the `scripts` directory, the following tools are found:

| Scripts Directory | Contains                                                       |
|-------------------|----------------------------------------------------------------|
| classifier        | classify instantaneous radiation measurements                  |
| cloud_types       | classify satellite into cloud types and convert to time series |
| event_stats       | event detection of classes and derived event statistics        |
| group_stats       | generate daily or hourly statistics for radiation data         |
| processor         | process 1-sec BSRN irradiance datasets                         |
| timeseries        | collection of some 1D data plots (timeseries)                  |
