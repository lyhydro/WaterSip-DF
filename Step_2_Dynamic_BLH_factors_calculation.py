# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from main_functions import get_and_combine_obs_files, get_files, back_tracking_files, readpartposit_to_df, \
    global_gridcell_info, midpoint, write_to_nc_2d
from concurrent.futures import ThreadPoolExecutor, as_completed
from YAMLConfig import YAMLConfig


# %% * * * * * * * * * * * * * * * * * * * * * MAIN * * * * * * * * * * * * * * * * * * * * *
def dynamic_blh_factors_calculation():
    print('begin step 2:dynamic BLH factors calculation !')
    config = YAMLConfig('config.yaml')
    general_config = config.get('General')
    tracking_days = general_config['tracking_days']
    time_span = general_config['time_span']
    output_spatial_resolution = general_config['output_spatial_resolution']
    start_time = str(general_config['start_time'])
    end_time = str(general_config['end_time'])

    warerSip_HAMSTER_config = config.get('WaterSip-HAMSTER')
    q_diff_e = warerSip_HAMSTER_config['q_diff_e']

    partposit_path = general_config['partposit_path']
    observation_path = warerSip_HAMSTER_config['observation_path']
    DF_file_path = config.get('WaterSip-DF')['DF_file_path']

    time = pd.date_range(start=pd.to_datetime(start_time, format='%Y%m%d%H') - pd.Timedelta(days=tracking_days)
                         , end=pd.to_datetime(end_time, format='%Y%m%d%H'), freq='{}H'.format(time_span))[:-1]

    ds = get_and_combine_obs_files(observation_path, time.strftime('%Y%m').drop_duplicates(), variable='e')
    E_era5_6h = -ds.resample(time='6H').sum(dim='time') * 1000  # hourly to 6-hourly, m to mm

    files0 = back_tracking_files(partposit_path, start_time, tracking_days, time_span)
    files = get_files(partposit_path, start_time, end_time, time_span)
    files = files0[:-1] + files
    latitude, longitude, gridcell_area = global_gridcell_info(output_spatial_resolution, lat_nor=90, lat_sou=-90,
                                                              lon_lef=-179, lon_rig=180)

    E_era5_6h = E_era5_6h.where(E_era5_6h > 0, 0)  # make e<0 to 0
    E_era5_6h = E_era5_6h.sel(time=time) * gridcell_area  # mm to kg
    DF_BLH_factors = np.zeros([np.shape(gridcell_area)[0], np.shape(gridcell_area)[1]])

    with ThreadPoolExecutor() as executor:
        future_to_index = {
            executor.submit(generate_files, i, files, q_diff_e, latitude, longitude, E_era5_6h, time,
                            output_spatial_resolution, DF_file_path): i
            for i in range(1, len(files))
        }
        for future in as_completed(future_to_index):
            i = future_to_index[future]
            try:
                future.result()
            except Exception as e:
                print(f'generate_files generated an exception: {e}')


def generate_files(i, files, q_diff_e, latitude, longitude, E_era5_6h, time, output_spatial_resolution, DF_file_path):
    # read end-time partposit
    df = readpartposit_to_df(files[i], variables=['lon', 'lat', 'q', 'z', 'blh', 'mass'])
    # read one-time-step before partposit
    df0 = readpartposit_to_df(files[i - 1], variables=['lon', 'lat', 'q', 'z', 'blh'])
    df0 = df0.rename(columns={'lat': 'lat0', 'lon': 'lon0', 'q': 'q0', 'z': 'z0', 'blh': 'blh0'})
    #
    df = df.merge(df0, left_index=True, right_index=True)
    # q_diff threshold filter
    df['q_diff'] = df['q'] - df['q0']
    df = df[df['q_diff'] > q_diff_e]
    df = df.drop(columns=['q', 'q0'])
    # calculate middle-lat and middle-lon
    df['lat'], df['lon'] = midpoint(df['lat'].values, df['lon'].values, df['lat0'].values, df['lon0'].values)
    df = df.drop(columns=['lat0', 'lon0'])
    # z and blh middle
    df['z'] = (df['z'] + df['z0']) / 2
    df['blh'] = (df['blh'] + df['blh0']) / 2
    df = df.drop(columns=['z0', 'blh0'])
    #
    df['lat_round'] = (df['lat'] / output_spatial_resolution).round() * output_spatial_resolution
    df['lon_round'] = (df['lon'] / output_spatial_resolution).round() * output_spatial_resolution
    df = df.drop(columns=['lon', 'lat'])
    # df = df[(df['lon_round'] <= 180) & (df['lon_round'] >= -179)]
    # calculate the dynamic BLH scaling factors
    BLH_factors_span = np.arange(0.2, 5.0, 0.02)  # set the smallest scaling factors step 0.02
    e_simulation = np.zeros([len(BLH_factors_span), len(latitude), len(longitude)])
    for j in range(0, len(BLH_factors_span)):
        df0 = df.copy()
        df0 = df0[df0['z'] < BLH_factors_span[j] * df0['blh']]
        # calculate e_mass
        df0['e_mass'] = df0['mass'] * df0['q_diff']  # kg
        df_grouped_e = df0.groupby(['lat_round', 'lon_round'])['e_mass'].sum().reset_index()
        # e写入0矩阵
        lat_idx_e = ((latitude[-1] - df_grouped_e['lat_round']) / output_spatial_resolution).astype(int)
        lon_idx_e = ((df_grouped_e['lon_round'] - longitude[0]) / output_spatial_resolution).astype(int)
        e_simulation[j, lat_idx_e, lon_idx_e] = df_grouped_e['e_mass']
        print(f"{BLH_factors_span[j]:.2f}", '- ', end='')

    obs0 = E_era5_6h[i - 1].values
    min_indices = np.nanargmin(np.abs(e_simulation - obs0), axis=0)
    # if all RH thresholds have no P, thus RH threshold is nan
    DF_BLH_factors = np.where(np.all(e_simulation == 0, axis=0), np.nan, BLH_factors_span[min_indices])
    DF_BLH_factors[obs0 == 0] = 0  # if no E, BLH_factors equals 0
    write_to_nc_2d(DF_BLH_factors, 'DF_BLH_factors', 'DF_BLH_factors_{}'.format(time[i - 1].strftime('%Y%m%d%H')),
                   latitude, longitude, DF_file_path)

    print(files[i - 1].split("partposit_")[-1], "done ！！！")
