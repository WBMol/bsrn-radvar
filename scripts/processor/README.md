# BSRN Cabauw Dataset Preprocessing
Scripts for downloading and (pre)processing the Baseline Surface Radiation Network (BSRN) data of Cabauw, NL.
More information on the measurement project can be found on the [KNMI website](https://projects.knmi.nl/cabauw/bsrn/).

## Processing steps
The starting point here is pre-processed raw instrument files.

Data processing consists of the following steps:
1. Add derived variables (solar angles, dhi)
2. Add McClear (clear-sky ghi, aod, tco3, tcwv)
3. Add Cabauw tower profiles (wind speed, wind direction)
4. Add automated quality filter for 1 Hz

### Solar Position
Note that pysolar should be installed with at the latest commit if you encounter an installation bug. 
They have, at the time of writing this, not yet marked a new release. 
Use the following syntax:

```bash
pip install git+git://github.com/pingswept/pysolar.git@b539efbd1c45bfac5ea51f8ee1f5f101cbb69dac
```

Shortwave beam (direct) irradiance is measured, but often it is useful to have the component normal to the surface.
For this, we need the solar evelation angle or we substract the diffuse from global irradiance.
Both are used, as a quality check, and because the latter can result in negative irradiance values near sunset/sunrise.

Direct irradiance normal to the surface is a simple calculation once you have the solar elevation angle `alpha`: `sw_dh = sw_beam * sin(alpha)`.
Calculating the solar elevation angle is done using `pysolar`, of which there are three settings:

| setting      | pysolar method      | temporal resolution | speed     | accuracy                                      |
|--------------|---------------------|---------------------|-----------|-----------------------------------------------|
| fast         | `get_altitude_fast` | 1 minute            | fast      | fine, but not good enough for quality control |
| accurate     | `get_altitude`      | 1 minute            | slow      | good                                          |
| accurate_max | `get_altitude`      | 1 second            | very slow | best                                          |

The fast and accurate setting are calculated every 1 minute, but interpolated to the resolution of the dataset.
The setting with which the data was prepared is included as an attribute.
The 'accurate'  setting is good enough for most use cases, whereas the 'max_accuracy' has little return value and is meant for debugging.
Note that 'max_accuracy' samples sub-60 seconds, but that only applies to 1-sec irradiance data.
### Clear-sky Irradiance

Clear-sky GHI and DHI are added to the daily datasets at a 1-minute temporal resolution for the Cabauw location. 
The data comes from the CAMS McClear dataset.
This is a reanalysis dataset including atmospheric profiles of e.g. aerosols and water content, 
greatly improving upon clear-sky estimates based on climatological atmospheric values or solar position alone.
Atmospheric properties are updated every 3 hours, this is unfortunately visible in the DHI, but the GHI seems to be very consistent.

More information on the McClear dataset [can be found on their website](http://www.soda-pro.com/web-services/radiation/cams-mcclear).

### Variable description
Below is a table with a description of the radiation and related variables in the processed dataset.
Variables coming from the raw BSRN data is given a more technical description, as the netCDF files lack such metadata.
The information in this table is also visible in the code, see `src/settings.py`.

| Variable                          | units | measurement interval (s) | measurement resolution | processed data netcdf name |
|-----------------------------------|-------|:------------------------:|------------------------|----------------------------|
| **BSRN**                          |       |                          |                        |                            |
| Shortwave Global                  | W/m2  |            1             | ~0.1                   | ghi                        |
| Shortwave Diffuse                 | W/m2  |            1             | ~0.1                   | dif                        |
| Shortwave Direct                  | W/m2  |            1             | ~0.1                   | dni                        |
| Longwave down*                    | W/m2  |            1             | <0.1                   | lwd                        |
| Air Temperature                   | C     |            12            | 0.1                    | t                          |
| Relative Humidity                 | %     |            12            | 1                      | rh                         |
| Pressure (relative)               | hPa   |            12            | 0.1                    | pmsl                       |
| **BSRN (derived)**                |       |                          |                        |                            |
| Shortwave Direct Horizontal (gmd) | W/m2  |            1             | ~0.1                   | dhi_gmd                    |
| Shortwave Direct Horizontal (sac) | W/m2  |            1             | ~0.1                   | dhi_sac                    |
| **McClear**                       |       |                          |                        |                            |
| Clear-sky GHI                     | W/m2  |            60            | -                      | ghi_cs                     |
| Clear-sky DIF                     | W/m2  |            60            | -                      | dif_cs                     |
