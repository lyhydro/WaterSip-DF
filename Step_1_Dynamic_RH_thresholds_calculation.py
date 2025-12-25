# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import threading
from functions import (get_and_combine_obs_files, get_files, readpartposit_to_df, 
                      global_gridcell_info, calculate_RH, midpoint, write_to_nc_2d, 
                      calculate_coordinate_round, get_algorithm_params)
from concurrent.futures import ThreadPoolExecutor, as_completed
import config

MAX_WORKERS = min(4, os.cpu_count())

file_write_lock = threading.Lock()


def dynamic_rh_thresholds_calculation(method):

    print(f'\nStep 1 ({method}): Calculate dynamic RH thresholds...')
    
    params = get_algorithm_params(method)
    
    time = pd.date_range(start=pd.to_datetime(config.START_TIME, format='%Y%m%d%H'),
                         end=pd.to_datetime(config.END_TIME, format='%Y%m%d%H'), 
                         freq='{}h'.format(config.TIME_SPAN))[:-1]

    ds = get_and_combine_obs_files(config.OBSERVATION_PATH, 
                                   time.strftime('%Y%m').drop_duplicates(), 
                                   variable='tp')
    P_era5_6h = ds.resample(time='6h').sum(dim='time') * 1000

    files = get_files(config.PARTPOSIT_PATH, config.START_TIME, config.END_TIME, config.TIME_SPAN)
    
    latitude, longitude, gridcell_area = global_gridcell_info(config.OUTPUT_SPATIAL_RESOLUTION, 
                                                               lat_nor=90, lat_sou=-90,
                                                               lon_lef=-179, lon_rig=180)

    P_era5_6h = P_era5_6h.where(P_era5_6h > 0, 0)
    P_era5_6h = P_era5_6h.sel(time=time) * gridcell_area

    def generate_files(i, files, latitude, longitude, P_era5_6h, time):

        df = readpartposit_to_df(files[i], variables=['lat', 'lon', 'q', 't', 'dens', 'mass'])
        calculate_RH(df)
        df = df.drop(columns=['t', 'dens'])
        
        df0 = readpartposit_to_df(files[i - 1], variables=['lat', 'lon', 'q', 't', 'dens'])
        df0 = df0.rename(columns={'lat': 'lat0', 'lon': 'lon0', 'q': 'q0', 't': 't0', 'dens': 'dens0'})
        calculate_RH(df0, RH_name='RH0', dens='dens0', q='q0', t='t0')
        df0 = df0.drop(columns=['t0', 'dens0'])
        
        df = df.merge(df0, left_index=True, right_index=True) 
        
        df['q_diff'] = df['q'] - df['q0']
        df = df[df['q_diff'] < params['q_diff_p']]
        df = df.drop(columns=['q', 'q0'])
        
        df['lat'], df['lon'] = midpoint(df['lat'].values, df['lon'].values, 
                                        df['lat0'].values, df['lon0'].values)
        df = df.drop(columns=['lat0', 'lon0'])
        
        df['RH'] = (df['RH'] + df['RH0']) / 2
        df = df.drop(columns=['RH0'])
        
        df['lat_round'] = calculate_coordinate_round(df['lat'], config.OUTPUT_SPATIAL_RESOLUTION)
        df['lon_round'] = calculate_coordinate_round(df['lon'], config.OUTPUT_SPATIAL_RESOLUTION)
        df = df.drop(columns=['lon', 'lat'])
        
        RH_span = np.arange(0, 100, 1)
        p_simulation = np.zeros([len(RH_span), len(latitude), len(longitude)])
        
        for j in range(0, len(RH_span)):
            df0 = df.copy()
            df0 = df0[df0['RH'] > RH_span[j]]
            df0['p_mass'] = -df0['mass'] * df0['q_diff']
            df_grouped_p = df0.groupby(['lat_round', 'lon_round'])['p_mass'].sum().reset_index()
            # sum P on grids
            lat_idx_p = ((latitude[0] - df_grouped_p['lat_round']) / config.OUTPUT_SPATIAL_RESOLUTION).astype(int)
            lon_idx_p = ((df_grouped_p['lon_round'] - longitude[0]) / config.OUTPUT_SPATIAL_RESOLUTION).astype(int)
            p_simulation[j, lat_idx_p, lon_idx_p] = df_grouped_p['p_mass']
        
        obs0 = P_era5_6h[i - 1].values
        min_indices = np.nanargmin(np.abs(p_simulation - obs0), axis=0)
        
        DF_RH_thresholds = np.where(np.all(p_simulation == 0, axis=0), np.nan, RH_span[min_indices])
        DF_RH_thresholds[obs0 == 0] = 100 
        
        with file_write_lock:
            write_to_nc_2d(DF_RH_thresholds, 'DF_RH_thresholds',
                           'DF_RH_thresholds_{}'.format(time[i - 1].strftime('%Y%m%d%H')), 
                           latitude, longitude, config.DF_FILE_PATH)
        
        del df, df0, p_simulation, obs0, DF_RH_thresholds

    print(f'Processing {len(files)-1} times with {MAX_WORKERS} workers...')
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_index = {
            executor.submit(generate_files, i, files, latitude, longitude, P_era5_6h, time): i
            for i in range(1, len(files))
        }
        
        for future in as_completed(future_to_index):
            future.result()

    print('Step 1 done!')


if __name__ == "__main__":
    dynamic_rh_thresholds_calculation('WaterSip-DF-HAMSTER') 
