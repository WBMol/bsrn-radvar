import os
import general.settings as gsettings

VERSION = '0.9'
debug = True
solar_elevation_setting = 'accurate'  # 'fast', 'accurate' or 'max_accuracy'
fastdir = True  # alternative tmp processing dir, usually ssd vs. hdd choice for speed

fdir_data = '/home/wouter/Work/data_tmp' if fastdir else gsettings.fdir_research_data

fdir_raw_1min = os.path.join(gsettings.fdir_research_data, 'Cabauw', 'BSRN', '1min', 'raw', '{y}')
fdir_out = os.path.join(fdir_data, 'Cabauw', 'BSRN', '{res}', 'processed', '{y}', '{m:02d}')
fdir_mcc = os.path.join(gsettings.fdir_research_data, 'McClear', 'NL')
fdir_ctower = os.path.join(gsettings.fdir_research_data, 'Cabauw', 'Tower', '{y}')
fdir_liaise = os.path.join(gsettings.fdir_research_data, 'Fieldwork', 'LIAISE', 'lacendrosa_radsoil', 'processed')

fdir_misc = './misc'
fdir_images = './images/'

ftp_url = 'bbc.knmi.nl'
ftp_user = os.environ.get('bbc_ftp_user')
ftp_pass = os.environ.get('bbc_ftp_pass')

bsrn_vars = {
    # raw radiation (related) variables
    'shortwave_global': {
        'raw_name': 'XQ1-T22CM22',
        'raw_dim': 'XQ1',
        'pangaea_name': 'SWD',
        'bsrn_nc_name': 'SWD',
        'raw_nc_name': 'SWD',
        'name': 'ghi',
        'dim': 'time_rad',
        'long_name': 'Global horizontal shortwave irradiance ',
        'units': 'W/m2'
    },
    'shortwave_diffuse': {
        'raw_name': 'XQ2-S04CM22',
        'raw_dim': 'XQ2',
        'pangaea_name': 'DIF',
        'bsrn_nc_name': 'DIF',
        'raw_nc_name': 'DIF',
        'name': 'dif',
        'dim': 'time_rad',
        'long_name': 'Diffuse horizontal shortwave irradiance',
        'units': 'W/m2'
    },
    'shortwave_direct': {
        'raw_name': 'XQ2-S04CH1_CA',
        'raw_dim': 'XQ2',
        'pangaea_name': 'DIR',
        'bsrn_nc_name': 'DIR',
        'raw_nc_name': 'DIR',
        'name': 'dni',
        'dim': 'time_rad',
        'long_name': 'Shortwave direct normal (to sun) irradiance',
        'units': 'W/m2'
    },
    'longwave_down': {
        'raw_name': 'XQ2-S04CG4',
        'raw_dim': 'XQ2',
        'pangaea_name': 'LWD',
        'bsrn_nc_name': 'DL',
        'raw_nc_name': 'LWD',
        'name': 'lwd',
        'dim': 'time_rad',
        'long_name': 'Longwave downward irradiance',
        'units': 'W/m2'
    },
    'temperature_body': {
        'raw_name': 'XT2-S04CH1_CA',
        'raw_dim': 'XT2',
        'name': 'tb',
        'dim': 'time_rad',
        'long_name': 'Body Temperature',
        'units': 'K'
    },
    'longwave_up': {
        'raw_nc_name': 'LWU',
        'name': 'lwu',
        'dim': 'time_rad',
        'long_name': 'Longwave upward irradiance',
        'units': 'W/m2'
    },
    'shortwave_up': {
        'raw_nc_name': 'SWU',
        'name': 'swu',
        'dim': 'time_rad',
        'long_name': 'Shortwave upward irradiance',
        'units': 'W/m2'
    },

    # derived radiation variables
    'shortwave_direct_horizontal_gmd': {
        'name': 'dhi_gmd',
        'dim': 'time_rad',
        'long_name': 'Direct horizontal irradiance (GHI minus DIF)',
        'units': 'W/m2'
    },
    'shortwave_direct_horizontal_sac': {
        'name': 'dhi_sac',
        'dim': 'time_rad',
        'long_name': 'Direct horizontal irradiance (solar angle corrected DNI)',
        'units': 'W/m2'
    },

    # raw AWS variables
    'temperature_air': {
        'raw_name': 'TX1-TA',
        'raw_dim': 'TX1',
        'pangaea_name': 'T2',
        'bsrn_nc_name': 'TEMP',
        'raw_nc_name': 'TEMP',
        'name': 't',
        'dim': 'time_aws',
        'long_name': '2m air temperature',
        'units': 'C'

    },
    'relative_humidity': {
        'raw_name': 'TX1-RH',
        'raw_dim': 'TX1',
        'pangaea_name': 'RH',
        'bsrn_nc_name': 'RH',
        'raw_nc_name': 'RH',
        'name': 'rh',
        'dim': 'time_aws',
        'long_name': 'Relative humidity',
        'units': '%'
    },
    'pressure': {
        'raw_name': 'PX1-PM',
        'raw_dim': 'PX1',
        'pangaea_name': 'PoPoPoPo',
        'bsrn_nc_name': 'PRES',
        'raw_nc_name': 'PRES',
        'name': 'pmsl',
        'dim': 'time_aws',
        'long_name': 'Pressure relative to mean sea level',
        'units': 'hPa'
    }
}

mcclear_vars = {
    # McClear variables http://www.soda-pro.com/web-services/radiation/cams-radiation-service
    'clear_sky_ghi': {
        'raw_name': 'CLEAR_SKY_GHI',
        'raw_dim': 'time',
        'name': 'ghi_cs',
        'dim': 'time_mcc',
        'long_name': 'Clear-sky global horizontal irradiance',
        'units': 'W/m2',
    },

    'clear_sky_dhi': {
        'raw_name': 'CLEAR_SKY_DHI',
        'raw_dim': 'time',
        'name': 'dif_cs',
        'dim': 'time_mcc',
        'long_name': 'Clear-sky diffuse horizontal irradiance',
        'units': 'W/m2'
    },

    'aod': {
        'raw_name': 'AOD',
        'raw_dim': 'datetime',
        'name': 'aod',
        'dim': 'time_mcc',
        'long_name': 'Aerosol optical depth',
        'description': 'CAMS McClear total AOD',
        'units': '-'
    },
    'tcwv': {
        'raw_name': 'tcwv',
        'raw_dim': 'datetime',
        'name': 'tcwv',
        'dim': 'time_mcc',
        'long_name': 'Total column water vapour',
        'description': 'CAMS McClear tcwv',
        'units': 'mm'
    },
    'tco3': {
        'raw_name': 'tco3',
        'raw_dim': 'datetime',
        'name': 'tco3',
        'dim': 'time_mcc',
        'long_name': 'Total column ozone',
        'description': 'CAMS McClear tco3',
        'units': 'Dobson units'
    },
}

sample_frequency = {
    # radiation sensor sample frequency is 1 Hz
    'time_rad': 'S',

    # automatic weather station sample frequency is 1/12 Hz, so once every 12 seconds
    'time_aws': '12S',

    # frequency of McClear clear-sky irradiances is once per minute
    'time_mcc': 'T',

    # frequency of McClear AOD is hourly
    'time_mccai': 'H'
}

station_latlons = {
    'cab': (51.968063, 4.927740)
}
