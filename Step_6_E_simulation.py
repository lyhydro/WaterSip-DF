# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import xarray as xr
from functions import (get_files, back_tracking_files, readpartposit_to_df, 
                      global_gridcell_info, midpoint, write_to_nc_2d, 
                      calculate_coordinate_round, get_algorithm_params)
import config


def e_simulation(method, df_method):

    print(f'Step 6 ({method}): E simulation...')
    
    params = get_algorithm_params(method)
    
    files0 = back_tracking_files(config.PARTPOSIT_PATH, config.START_TIME, config.TRACKING_DAYS, config.TIME_SPAN)
    files1 = get_files(config.PARTPOSIT_PATH, config.START_TIME, config.END_TIME, config.TIME_SPAN)
    files = files0[::-1] + files1[1:]  
    
    latitude, longitude, gridcell_area = global_gridcell_info(config.OUTPUT_SPATIAL_RESOLUTION, 
                                                              lat_nor=90, lat_sou=-90,
                                                              lon_lef=-179, lon_rig=180)
    
    temp_time = pd.date_range(start=pd.to_datetime(files[0][-14:-4], format='%Y%m%d%H'),
                             end=pd.to_datetime(files[-1][-14:-4], format='%Y%m%d%H'),
                             freq='{}h'.format(config.TIME_SPAN))[:-1]

    for i in range(1, len(files)):
        try:

            df_e = readpartposit_to_df(files[i], variables=['lon', 'lat', 'q', 'z', 'blh', 'mass'])
            
            df0 = readpartposit_to_df(files[i - 1], variables=['lon', 'lat', 'q', 'z', 'blh'])
            df0 = df0.rename(columns={'lat': 'lat0', 'lon': 'lon0', 'q': 'q0', 'z': 'z0', 'blh': 'blh0'})
            
            df_e = df_e.merge(df0, left_index=True, right_index=True)
            
            df_e['q_diff'] = df_e['q'] - df_e['q0']
            df_e = df_e.drop(columns=['q', 'q0'])
            df_e = df_e[df_e['q_diff'] > 0]  
            
            df_e['lat'], df_e['lon'] = midpoint(df_e['lat'].values, df_e['lon'].values, 
                                                df_e['lat0'].values, df_e['lon0'].values)
            df_e = df_e.drop(columns=['lat0', 'lon0'])
            
            df_e['z'] = (df_e['z'] + df_e['z0']) / 2
            df_e['blh'] = (df_e['blh'] + df_e['blh0']) / 2
            df_e = df_e.drop(columns=['z0', 'blh0'])
            
            df_e['lat_round'] = calculate_coordinate_round(df_e['lat'], config.OUTPUT_SPATIAL_RESOLUTION)
            df_e['lon_round'] = calculate_coordinate_round(df_e['lon'], config.OUTPUT_SPATIAL_RESOLUTION)
            df_e = df_e.drop(columns=['lon', 'lat'])
            
            if df_method == 'on':
                BLH_factor = xr.open_dataset(
                    f'{config.DF_FILE_PATH}/DF_BLH_factors_{temp_time[i - 1].strftime("%Y%m%d%H")}.nc'
                )['DF_BLH_factors']
                df_BLH_factor = BLH_factor.to_dataframe(name='BLH_factor').reset_index()
                df_BLH_factor = df_BLH_factor.query("BLH_factor != 0 and BLH_factor.notna()", engine='numexpr')
                
                df_e = pd.merge(df_e, df_BLH_factor, how='left', 
                               left_on=['lat_round', 'lon_round'],
                               right_on=['latitude', 'longitude'])
                df_e = df_e.drop(columns=['latitude', 'longitude'])
                df_e.dropna(subset=['BLH_factor'], inplace=True)
                df_e = df_e[df_e['z'] <= df_e['blh'] * df_e['BLH_factor']]
            else:
                df_e = df_e[df_e['q_diff'] > params['q_diff_e']]
                df_e = df_e[df_e['z'] <= df_e['blh'] * params['blh_factor']]
            
            df_e['e_mass'] = df_e['mass'] * df_e['q_diff']  # kg
            
            df_grouped_e = df_e.groupby(['lat_round', 'lon_round'])['e_mass'].sum().reset_index()
            
            E_simulation_mm = np.zeros([len(latitude), len(longitude)])
            
            if len(df_grouped_e) > 0:

                lat_idx_e = ((latitude[-1] - df_grouped_e['lat_round']) / config.OUTPUT_SPATIAL_RESOLUTION).astype(int)
                lon_idx_e = ((df_grouped_e['lon_round'] - longitude[0]) / config.OUTPUT_SPATIAL_RESOLUTION).astype(int)
                
                E_simulation_mm[lat_idx_e, lon_idx_e] = df_grouped_e['e_mass']
                
            E_simulation_mm = E_simulation_mm / gridcell_area
            
            write_to_nc_2d(E_simulation_mm, 'E_simulation_mm',
                           f'E_simulation_mm_{temp_time[i - 1].strftime("%Y%m%d%H")}', 
                           latitude, longitude, config.P_E_SIMULATION_OUTPUT_PATH)
            
            del df_e, df0, df_grouped_e, E_simulation_mm
            
        except Exception as e:
            print(f"! Error processing file {files[i]}: {e}")
            continue

    print('Step 6 done!')


if __name__ == "__main__":
    e_simulation('WaterSip-DF-HAMSTER', 'on') 