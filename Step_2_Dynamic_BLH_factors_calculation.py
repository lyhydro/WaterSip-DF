# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import threading
from functions import (get_and_combine_obs_files, get_files, readpartposit_to_df, 
                      global_gridcell_info, midpoint, write_to_nc_2d, 
                      calculate_coordinate_round, get_algorithm_params, back_tracking_files)
from concurrent.futures import ThreadPoolExecutor, as_completed
import config

MAX_WORKERS = min(4, os.cpu_count())

file_write_lock = threading.Lock()


def dynamic_blh_factors_calculation(method):

    print(f'\nStep 2 ({method}): Calculate dynamic BLH factors...')
    
    params = get_algorithm_params(method)
    
    time = pd.date_range(
        start=pd.to_datetime(config.START_TIME, format='%Y%m%d%H') - pd.Timedelta(days=config.TRACKING_DAYS),
        end=pd.to_datetime(config.END_TIME, format='%Y%m%d%H'), 
        freq='{}h'.format(config.TIME_SPAN))[:-1]

    ds = get_and_combine_obs_files(config.OBSERVATION_PATH, 
                                   time.strftime('%Y%m').drop_duplicates(), 
                                   variable='e')
    E_era5_6h = -ds.resample(time='6h').sum(dim='time') * 1000

    files0 = back_tracking_files(config.PARTPOSIT_PATH, config.START_TIME, config.TRACKING_DAYS, config.TIME_SPAN)
    files1 = get_files(config.PARTPOSIT_PATH, config.START_TIME, config.END_TIME, config.TIME_SPAN)
    files = files0[::-1] + files1[1:]
    
    latitude, longitude, gridcell_area = global_gridcell_info(config.OUTPUT_SPATIAL_RESOLUTION, 
                                                               lat_nor=90, lat_sou=-90,
                                                               lon_lef=-179, lon_rig=180)

    E_era5_6h = E_era5_6h.where(E_era5_6h > 0, 0)
    E_era5_6h = E_era5_6h.sel(time=time) * gridcell_area

    def generate_files(i, files, latitude, longitude, E_era5_6h, time):

        df = readpartposit_to_df(files[i], variables=['lat', 'lon', 'q', 'z', 'blh', 'mass'])
        
        df0 = readpartposit_to_df(files[i - 1], variables=['lat', 'lon', 'q', 'z', 'blh'])
        df0 = df0.rename(columns={'lat': 'lat0', 'lon': 'lon0', 'q': 'q0', 'z': 'z0', 'blh': 'blh0'})
        
        df = df.merge(df0, left_index=True, right_index=True) 
        
        df['q_diff'] = df['q'] - df['q0']
        df = df[df['q_diff'] > params['q_diff_e']]
        df = df.drop(columns=['q', 'q0'])
        
        df['lat'], df['lon'] = midpoint(df['lat'].values, df['lon'].values, 
                                        df['lat0'].values, df['lon0'].values)
        df = df.drop(columns=['lat0', 'lon0'])
        
        df['z'] = (df['z'] + df['z0']) / 2
        df['blh'] = (df['blh'] + df['blh0']) / 2
        df = df.drop(columns=['z0', 'blh0'])
        
        df['lat_round'] = calculate_coordinate_round(df['lat'], config.OUTPUT_SPATIAL_RESOLUTION)
        df['lon_round'] = calculate_coordinate_round(df['lon'], config.OUTPUT_SPATIAL_RESOLUTION)
        df = df.drop(columns=['lon', 'lat'])
        
        # test the dynamic BLH_factors
        # BLH_factors_span = np.concatenate([np.arange(0, 5, 0.05), 
        #                                    np.arange(5, 10, 0.5), 
        #                                    np.arange(10, 100, 10), 
        #                                    np.arange(100, 1000, 100), 
        #                                    np.arange(1000, 10001, 1000)])        
        BLH_factors_span = np.concatenate([np.arange(0, 3, 0.05), 
                                           np.arange(3, 6, 0.1), 
                                           np.arange(6, 10, 0.25)] )
               
        e_simulation = np.zeros([len(BLH_factors_span), len(latitude), len(longitude)])
        
        for j in range(0, len(BLH_factors_span)):
            df0 = df.copy()
            df0 = df0[df0['z'] < BLH_factors_span[j] * df0['blh']]
            df0['e_mass'] = df0['mass'] * df0['q_diff']
            df_grouped_e = df0.groupby(['lat_round', 'lon_round'])['e_mass'].sum().reset_index()
            lat_idx_e = ((latitude[0] - df_grouped_e['lat_round']) / config.OUTPUT_SPATIAL_RESOLUTION).astype(int)
            lon_idx_e = ((df_grouped_e['lon_round'] - longitude[0]) / config.OUTPUT_SPATIAL_RESOLUTION).astype(int)
            e_simulation[j, lat_idx_e, lon_idx_e] = df_grouped_e['e_mass']
        
        obs0 = E_era5_6h[i - 1].values
        min_indices = np.nanargmin(np.abs(e_simulation - obs0), axis=0)
        DF_BLH_factors = np.where(np.all(e_simulation == 0, axis=0), np.nan, BLH_factors_span[min_indices])
        DF_BLH_factors[obs0 == 0] = 0
        
        with file_write_lock:
            write_to_nc_2d(DF_BLH_factors, 'DF_BLH_factors',
                           'DF_BLH_factors_{}'.format(time[i - 1].strftime('%Y%m%d%H')), 
                           latitude, longitude, config.DF_FILE_PATH)
        
        del df, df0, e_simulation, obs0, DF_BLH_factors

    print(f'Processing {len(files)-1} times with {MAX_WORKERS} workers...')
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_index = {
            executor.submit(generate_files, i, files, latitude, longitude, E_era5_6h, time): i
            for i in range(1, len(files))
        }
        
        for future in as_completed(future_to_index):
            future.result()

    print('Step 2 done!')


if __name__ == "__main__":
    dynamic_blh_factors_calculation('WaterSip-DF-HAMSTER') 