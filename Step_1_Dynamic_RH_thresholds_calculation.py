# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from main_functions import get_and_combine_obs_files, get_files, readpartposit_to_df, global_gridcell_info, \
    calculate_RH, midpoint, write_to_nc_2d
from concurrent.futures import ThreadPoolExecutor, as_completed
from YAMLConfig import YAMLConfig


# %% * * * * * * * * * * * * * * * * * * * * * MAIN * * * * * * * * * * * * * * * * * * * * *
def dynamic_rh_thresholds_calculation():
    print('begin step 1:dynamic RH thresholds calculation !')
    config = YAMLConfig('config.yaml')
    general_config = config.get('general')
    time_span = general_config['time_span']
    start_time = str(general_config['start_time'])
    end_time = str(general_config['end_time'])

    output_spatial_resolution = general_config['output_spatial_resolution']

    warerSip_HAMSTER_config = config.get('warerSip-HAMSTER')
    # threshold used to extract all potential E particles
    q_diff_p = warerSip_HAMSTER_config['q_diff_p']

    partposit_path = general_config['partposit_path']
    observation_path = warerSip_HAMSTER_config['observation_path']
    DF_file_path = config.get('warerSip-DF')['DF_file_path']

    time = pd.date_range(start=pd.to_datetime(start_time, format='%Y%m%d%H'),
                         end=pd.to_datetime(end_time, format='%Y%m%d%H'), freq='{}H'.format(time_span))[:-1]

    ds = get_and_combine_obs_files(observation_path, time.strftime('%Y%m').drop_duplicates(), variable='tp')
    P_era5_6h = ds.resample(time='6H').sum(dim='time') * 1000  # hourly to 6-hourly, m to mm

    files = get_files(partposit_path, start_time, end_time, time_span)
    latitude, longitude, gridcell_area = global_gridcell_info(output_spatial_resolution, lat_nor=90, lat_sou=-90,
                                                              lon_lef=-179, lon_rig=180)

    P_era5_6h = P_era5_6h.where(P_era5_6h > 0, 0)  # make p<0 to 0
    P_era5_6h = P_era5_6h.sel(time=time) * gridcell_area  # mm to kg
    DF_RH_thresholds = np.zeros([np.shape(gridcell_area)[0], np.shape(gridcell_area)[1]])

    with ThreadPoolExecutor() as executor:
        future_to_index = {
            executor.submit(generate_files, i, files, q_diff_p, latitude, longitude, P_era5_6h, time,
                            output_spatial_resolution, DF_file_path): i
            for i in range(1, len(files))
        }
        for future in as_completed(future_to_index):
            i = future_to_index[future]
            try:
                future.result()
            except Exception as exc:
                print(f'generate_files generated an exception: {exc}')


def generate_files(i, files, q_diff_p, latitude, longitude, P_era5_6h, time, output_spatial_resolution, DF_file_path):
    # read end-time partposit
    df = readpartposit_to_df(files[i], variables=['lon', 'lat', 'q', 't', 'dens', 'mass'])
    calculate_RH(df)
    df = df.drop(columns=['t', 'dens'])
    # read one-time-step before partposit
    df0 = readpartposit_to_df(files[i - 1], variables=['lon', 'lat', 'q', 't', 'dens'])
    df0 = df0.rename(columns={'lat': 'lat0', 'lon': 'lon0', 'q': 'q0', 't': 't0', 'dens': 'dens0'})
    calculate_RH(df0, RH_name='RH0', dens='dens0', q='q0', t='t0')
    df0 = df0.drop(columns=['t0', 'dens0'])
    #
    df = df.merge(df0, left_index=True, right_index=True)
    # q_diff filtering
    df['q_diff'] = df['q'] - df['q0']
    df = df[df['q_diff'] < q_diff_p]
    df = df.drop(columns=['q', 'q0'])
    # calculate middle-lat and middle-lon
    df['lat'], df['lon'] = midpoint(df['lat'].values, df['lon'].values, df['lat0'].values, df['lon0'].values)
    df = df.drop(columns=['lat0', 'lon0'])
    # mid_RH
    df['RH'] = (df['RH'] + df['RH0']) / 2
    df = df.drop(columns=['RH0'])
    #
    df['lat_round'] = (df['lat'] / output_spatial_resolution).round() * output_spatial_resolution
    df['lon_round'] = (df['lon'] / output_spatial_resolution).round() * output_spatial_resolution
    df = df.drop(columns=['lon', 'lat'])
    # df = df[(df['lon_round'] <= 180) & (df['lon_round'] >= -179)]
    # calculate the dynamic RH thresholds
    RH_span = np.arange(0, 100.1, 0.2)  # set the smallest RH thresholds step 0.2
    p_simulation = np.zeros([len(RH_span), len(latitude), len(longitude)])
    for j in range(0, len(RH_span)):
        df0 = df.copy()
        df0 = df0[df0['RH'] > RH_span[j]]
        # calculate p_mass
        df0['p_mass'] = -df0['mass'] * df0['q_diff']  # kg, *-1 makes P positive
        # sum p on grids
        df_grouped_p = df0.groupby(['lat_round', 'lon_round'])['p_mass'].sum().reset_index()
        # p to martix
        lat_idx_p = ((latitude[-1] - df_grouped_p['lat_round']) / output_spatial_resolution).astype(int)
        lon_idx_p = ((df_grouped_p['lon_round'] - longitude[0]) / output_spatial_resolution).astype(int)
        p_simulation[j, lat_idx_p, lon_idx_p] = df_grouped_p['p_mass']
        print(f"{RH_span[j]:.1f}", '- ', end='')
    obs0 = P_era5_6h[i - 1].values
    min_indices = np.nanargmin(np.abs(p_simulation - obs0), axis=0)
    # if all RH thresholds have no P, thus RH threshold is nan
    DF_RH_thresholds = np.where(np.all(p_simulation == 0, axis=0), np.nan, RH_span[min_indices])
    DF_RH_thresholds[obs0 == 0] = 100  # if no P, RH threshold equals 100
    write_to_nc_2d(DF_RH_thresholds, 'DF_RH_thresholds',
                   'DF_RH_thresholds_{}'.format(time[i - 1].strftime('%Y%m%d%H')), latitude, longitude,
                   DF_file_path)
    print(files[i - 1].split("partposit_")[-1], "done ！！！")
