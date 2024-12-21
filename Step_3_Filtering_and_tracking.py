# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
import xarray as xr
from main_functions import get_files, readpartposit_to_df, from_shp_get_parts, from_bounds_get_parts, calculate_RH, midpoint, \
    back_tracking_files, get_df_BLH_factor, writepkl
import time
from YAMLConfig import YAMLConfig


# %% * * * * * * * * * * * * * * * * * * * * * MAIN - found all P particles * * * * * * * * * * * * * * * * * * * * *
def filtering_and_tracking(df_method):
    begin_time = time.time()
    print('begin step 3:filtering process !')
    config = YAMLConfig('config.yaml')
    general_config = config.get('general')
    tracking_days = general_config['tracking_days']
    time_span = general_config['time_span']
    output_spatial_resolution = general_config['output_spatial_resolution']
    start_time = str(general_config['start_time'])
    end_time = str(general_config['end_time'])

    warerSip_HAMSTER_config = config.get('warerSip-HAMSTER')
    q_diff_p = warerSip_HAMSTER_config['q_diff_p']
    q_diff_e = warerSip_HAMSTER_config['q_diff_e']

    watersip_config = config.get('watersip')
    default_q_diff_p = watersip_config['default_q_diff_p']
    default_q_diff_e = watersip_config['default_q_diff_e']
    default_rh_threshold = watersip_config['default_rh_threshold']
    default_blh_factor = watersip_config['default_blh_factor']

    partposit_path = general_config['partposit_path']
    DF_file_path = config.get('warerSip-DF')['DF_file_path']
    target_region = general_config['target_region']
    temporary_file_path = general_config['temporary_file_path']

    print(f"start_time：{start_time}")
    print(f"end_time：{end_time}")
    files = get_files(partposit_path, start_time, end_time, time_span)
    # selecting all P particles in each time step in the shp defined target region
    all_P_parts = [];
    times = []
    for i in range(1, len(files)):  # begain from 1 to calculate q_diff
        # read partposit
        df = readpartposit_to_df(files[i], variables=['lon', 'lat', 'q', 't', 'dens', 'mass'])
        calculate_RH(df)
        # read one-time-step before partposit
        df0 = readpartposit_to_df(files[i - 1], variables=['lon', 'lat', 'q', 't', 'dens'])
        df0 = df0.rename(columns={'lat': 'lat0', 'lon': 'lon0', 'q': 'q0', 't': 't0', 'dens': 'dens0'})
        calculate_RH(df0, RH_name='RH0', dens='dens0', q='q0', t='t0')
        # merge
        df = df.merge(df0, left_index=True, right_index=True)
        df = df.drop(columns=['dens', 'dens0', 't', 't0'])
        # q_diff threshold filter
        df['q_diff'] = df['q'] - df['q0']
        df = df.drop(columns=['q', 'q0'])
        df = df[df['q_diff'] < q_diff_p]
        # middle lat and lon
        df['lat'], df['lon'] = midpoint(df['lat'].values, df['lon'].values, df['lat0'].values, df['lon0'].values)
        df = df.drop(columns=['lat0', 'lon0'])
        # 筛选研究区粒子
        if target_region[-4:]=='.shp':
            df = from_shp_get_parts(df, target_region)
        elif isinstance(target_region, list):
            df = from_bounds_get_parts(df, target_region[0], target_region[1], target_region[2], target_region[3])
        # mid_RH
        df['RH'] = (df['RH'] + df['RH0']) / 2
        df = df.drop(columns=['RH0'])
        ## first correct by rh_threshold
        tmp_time = files[i - 1][-14:-4]
        if df_method == 'on':
            df['lat_round'] = (df['lat'] / output_spatial_resolution).round() * output_spatial_resolution
            df['lon_round'] = (df['lon'] / output_spatial_resolution).round() * output_spatial_resolution
            # df = df[(df['lon_round'] <= 180) & (df['lon_round'] >= -179)] # obs coverage
            # get DF_RH_thresholds and transfor to dataframe
            RH_threshold = xr.open_dataset('{}\DF_RH_thresholds_{}.nc'.format(DF_file_path, tmp_time))[
                'DF_RH_thresholds']
            df_RH_threshold = RH_threshold.to_dataframe(name='RH_threshold').reset_index()
            df_RH_threshold = df_RH_threshold.query("RH_threshold != 100 and RH_threshold.notna()", engine='numexpr')
            #
            original_index = df.index  # keep index
            df = pd.merge(df, df_RH_threshold, how='left', left_on=['lat_round', 'lon_round'],
                          right_on=['latitude', 'longitude'])  ## slow !!!
            df.index = original_index  # recover index
            df = df[df['RH'] >= df['RH_threshold']]
            df = df.drop(columns=['latitude', 'longitude', 'RH', 'RH_threshold', 'lat_round', 'lon_round'])
        else:  # use default rh_threshold
            df = df[df['q_diff'] < default_q_diff_p]
            df = df[df['RH'] > default_rh_threshold]
            df = df.drop(columns=['RH'])
        # calculate p simulation
        df['p_mass'] = -df['mass'] * df['q_diff']
        all_P_parts.append(df)
        times.append(tmp_time)
        if len(df) == 0:
            print("!!!!! no parts found at this time-step !!!!!")
        print('selecting P particles in', tmp_time)

    # %% * * * * * * * * * * * * * * * * * * * * * MAIN - tracking processes * * * * * * * * * * * * * * * * * * * * *
    # get and store P particles backtrack files in each time-step, and establish tracking table for each particle
    for i in range(0, len(times)):
        track_particles(i, times, all_P_parts, temporary_file_path, tracking_days, time_span, partposit_path, output_spatial_resolution,
                        DF_file_path, df_method, q_diff_e, default_q_diff_e, default_blh_factor)
    print('temporary_file done!')
    end_time = time.time()
    execution_time = end_time - begin_time
    print(f"程序执行时间：{execution_time}秒")
    print('step 3 done!')
    # %%


def track_particles(i, times, all_P_parts, temporary_file_path, tracking_days, time_span, partposit_path,
                    output_spatial_resolution, DF_file_path, df_method, q_diff_e, default_q_diff_e, default_blh_factor):
    result = []
    files_track = back_tracking_files(partposit_path, times[i], tracking_days, time_span)
    result.append(all_P_parts[i])
    #
    for f in files_track[::-1]:
        df0 = readpartposit_to_df(f, variables=['lon', 'lat', 'z', 'q', 'blh', 'mass'])
        result.append(df0.loc[df0.index.intersection(result[-1].index)])  ### show !!!用result[-1].index从df0提取数据
        print('.', end='')
    writepkl("{}\{}-{} P particle backward files.pkl".format(temporary_file_path, times[i], time_span), result)
    print('\n', times[i], 'P particles get backtrack files done')
    ## establish particle tracking table
    indexs = result[0].index
    df0 = [];
    n = 1
    track_times = pd.to_datetime(times[i], format='%Y%m%d%H') - pd.Timedelta(hours=time_span) * np.arange(
        len(result) - 1)
    for index in indexs:
        # each particle has a dataframe table from tracking start to end
        df_index = []
        df_index.append(result[0].loc[index][['lon', 'lat', 'q_diff', 'mass', 'p_mass']])
        for j in range(1, len(result)):
            if index in result[j].index:
                df_index.append(result[j].loc[index][['lon', 'lat', 'z', 'q', 'blh']])
            else:
                df_index.append(df_index[-1].apply(lambda x: np.nan))  # if particle lost, set nan
        df_index = pd.DataFrame(df_index)
        # lat_mid, lon_mid, z_mid, q_diff, and abl_mid
        df_index['lat'][1:-1], df_index['lon'][1:-1] = midpoint(df_index['lat'][1:-1].values,
                                                                df_index['lon'][1:-1].values,
                                                                df_index['lat'][2:].values, df_index['lon'][2:].values)
        df_index['z'][1:-1] = (df_index['z'][1:-1].values + df_index['z'][2:].values) / 2
        df_index['q_diff'][1:-1] = df_index['q'][1:-1].values - df_index['q'][2:].values
        df_index['blh'][1:-1] = (df_index['blh'][1:-1].values + df_index['blh'][2:].values) / 2
        df_index = df_index.iloc[:-1]
        #
        if df_method == 'on':  # use dynamic ABL_factor
            df_index['lat_round'] = (df_index['lat'] / output_spatial_resolution).round() * output_spatial_resolution
            df_index['lon_round'] = (df_index['lon'] / output_spatial_resolution).round() * output_spatial_resolution
            df_index['start_time'] = track_times
            #
            df_BLH_factor = get_df_BLH_factor(DF_file_path, times[i], tracking_days, direction='backward')
            df_index = pd.merge(df_index, df_BLH_factor, how='left', left_on=['lat_round', 'lon_round', 'start_time'],
                                right_on=['latitude', 'longitude', 'time'])  ## show !!!
            df_index = df_index.drop(columns=['latitude', 'longitude', 'start_time', 'time', 'lat_round', 'lon_round'])
            # selecting E region, and calculate contribution ratio f in each time step
            df_index['f0'] = np.where(
                (df_index['q_diff'] > q_diff_e) & (df_index['z'] < (df_index['BLH_factor'] * df_index['blh'])),
                df_index['q_diff'] / df_index['q'], 0)
            df_index = df_index.drop(columns=['BLH_factor'])
        else:  # use default ABL_factor
            df_index['f0'] = np.where(
                (df_index['q_diff'] > default_q_diff_e) & (df_index['z'] < (default_blh_factor * df_index['blh'])),
                df_index['q_diff'] / df_index['q'], 0)
        df_index = df_index.drop(columns=['z', 'blh', 'q_diff', 'q'])
        # cycle to weight f0 to f
        df_index['f'] = df_index['f0']
        for l in range(2, len(df_index)):
            # df_index['f'].iloc[l:] = df_index['f'].iloc[l:]*(1-df_index['f0'].iloc[l-1])
            df_index.iloc[l:, df_index.columns.get_loc('f')] = df_index['f'].iloc[l:] * (1 - df_index['f0'].iloc[l - 1])
        df_index = df_index.drop(columns=['f0'])
        df0.append(df_index)
        print(times[i], 'establishing tracking table', n, '/', len(indexs))
        n = n + 1
    writepkl(os.path.join(temporary_file_path, "{}-{} {}.pkl".format(times[i], time_span, 'P particle tracking table')),
             df0)
    print('\n', times[i], 'P particles get backtrack files done')
