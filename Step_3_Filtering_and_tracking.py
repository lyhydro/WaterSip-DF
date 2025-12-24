# -*- coding: utf-8 -*-

import os
import pandas as pd
import polars as pl  
import numpy as np
import xarray as xr
import time
import gc
from functions import (get_files, readpartposit_to_df, from_shp_get_parts, 
                      from_bounds_get_parts, calculate_RH, midpoint, 
                      back_tracking_files, get_df_BLH_factor, writepkl, 
                      calculate_coordinate_round, get_algorithm_params)
import config

def filtering_and_tracking(method, df_method):

    begin_time = time.time()
    print(f'\nStep 3 ({method}): Particle filtering and moisture tracking...')
    
    params = get_algorithm_params(method)
    files_list = get_files(config.PARTPOSIT_PATH, config.START_TIME, config.END_TIME, config.TIME_SPAN)
    
    temp_rh_threshold = {}

    #%% ==========================================================================
    def get_and_track_particles(file, prefile, df_method, i):

        df = readpartposit_to_df(file, variables=['lon', 'lat', 'z', 'q', 'blh', 'mass', 't', 'dens'])
        df0 = readpartposit_to_df(prefile, variables=['lon', 'lat', 'z', 'q', 'blh', 't', 'dens'])
        
        if 'RH' not in df.columns:
            calculate_RH(df)
        if 'RH' not in df0.columns:
            calculate_RH(df0, RH_name='RH', dens='dens', q='q', t='t')
 
        common_indices = df.index.intersection(df0.index)        
        df = df.loc[common_indices]
        df0 = df0.loc[common_indices]        

        df_pl = pl.from_pandas(df.reset_index())
        df0_pl = pl.from_pandas(df0.reset_index()) 
        
        result_pl = df_pl.join(df0_pl, on="index", suffix="_0").with_columns([
            (pl.col("q") - pl.col("q_0")).alias("q_diff"),
            ((pl.col("RH") + pl.col("RH_0")) / 2).alias("RH_avg")
        ]).filter(
            pl.col("q_diff") < params['q_diff_p']
        ).select([
            "index", "lat", "lon", "lat_0", "lon_0", "q_diff", "RH_avg", "mass"
        ])
        
        if result_pl.height == 0:
            print(f"No particles after q_diff filtering for {prefile[-14:-4]}")
            return
        
        result_df = result_pl.to_pandas().set_index('index')
        result_df = result_df.rename(columns={'RH_avg': 'RH'})
        
        result_df['lat'], result_df['lon'] = midpoint(result_df['lat'].values, 
                                                      result_df['lon'].values, 
                                                      result_df['lat_0'].values, 
                                                      result_df['lon_0'].values)       
        result_df = result_df.drop(columns=['lat_0', 'lon_0'])
        
        if isinstance(config.TARGET_REGION, str) and config.TARGET_REGION.endswith('.shp'):
            result_df = from_shp_get_parts(result_df, config.TARGET_REGION)
        elif isinstance(config.TARGET_REGION, list):
            result_df = from_bounds_get_parts(result_df, 
                                              config.TARGET_REGION[0], 
                                              config.TARGET_REGION[1], 
                                              config.TARGET_REGION[2],
                                              config.TARGET_REGION[3])
        
        tmp_time = prefile[-14:-4]
        if df_method == 'on':
            result_df['lat_round'] = calculate_coordinate_round(result_df['lat'], config.OUTPUT_SPATIAL_RESOLUTION)
            result_df['lon_round'] = calculate_coordinate_round(result_df['lon'], config.OUTPUT_SPATIAL_RESOLUTION)
            
            if tmp_time not in temp_rh_threshold:
                rh_file = f'{config.DF_FILE_PATH}/DF_RH_thresholds_{tmp_time}.nc'
                RH_threshold = xr.open_dataset(rh_file)['DF_RH_thresholds']
                temp_rh_threshold[tmp_time] = RH_threshold
            else:
                RH_threshold = temp_rh_threshold[tmp_time]
            
            df_RH_threshold = RH_threshold.to_dataframe(name='RH_threshold').reset_index()
            df_RH_threshold = df_RH_threshold.query("RH_threshold != 100 and RH_threshold.notna()", engine='numexpr')
            
            result_pl = pl.from_pandas(result_df.reset_index())
            rh_threshold_pl = pl.from_pandas(df_RH_threshold)
            
            result_pl = result_pl.with_columns([
                pl.col("lat_round").cast(pl.Float64),
                pl.col("lon_round").cast(pl.Float64)
            ])
            rh_threshold_pl = rh_threshold_pl.with_columns([
                pl.col("latitude").cast(pl.Float64),
                pl.col("longitude").cast(pl.Float64)
            ])
            
            merged_pl = result_pl.join(
                rh_threshold_pl,
                left_on=['lat_round', 'lon_round'],
                right_on=['latitude', 'longitude'],
                how='left'
            ).filter(
                pl.col('RH') >= pl.col('RH_threshold')
            ).select([
                "index", "lat", "lon", "q_diff", "mass"
            ])
            
            result_df = merged_pl.to_pandas().set_index('index')
            
        else:
            rh_threshold = params.get('rh_threshold')
            result_df = result_df[result_df['RH'] > rh_threshold]
            result_df = result_df.drop(columns=['RH'])
        
        result_df['p_mass'] = -result_df['mass'] * result_df['q_diff']
        
        track_particles(i - 1, tmp_time, result_df, df_method)
        
        del result_df, result_pl, merged_pl  
        gc.collect()
    
    def track_particles(i, times, all_P_parts, df_method):
        print(f"\nStart track particles for {times}, particle number: {len(all_P_parts)}")
        
        if len(all_P_parts) == 0:
            print(f"! Warning: {times} No particles found")

            empty_result = []
            output_path = f"{config.TEMPORARY_FILE_PATH}/{times}-{config.TIME_SPAN}_P_particle_backward_files.pkl"
            writepkl(output_path, empty_result)
            
            empty_tracking_table = []
            tracking_output_path = os.path.join(config.TEMPORARY_FILE_PATH,
                                               f"{times}-{config.TIME_SPAN}_P_particle_tracking_table.pkl")
            writepkl(tracking_output_path, empty_tracking_table)
            print(f"Generated empty pkl files for {times}")
            return
            
        files_track = back_tracking_files(config.PARTPOSIT_PATH, times, config.TRACKING_DAYS, config.TIME_SPAN)
        
        result = [None] * (len(files_track) + 1)
        result[0] = all_P_parts
        
        particle_indices = all_P_parts.index.values
        
        print("Loading files with Polars acceleration...")
        for idx, f in enumerate(files_track):
            try:

                df_pd = readpartposit_to_df(f, variables=['lon', 'lat', 'z', 'q', 'blh', 'mass'])
                
                df_pl = pl.from_pandas(df_pd.reset_index())
                
                common_indices = np.intersect1d(particle_indices, df_pd.index.values)
                
                if len(common_indices) > 0:
                    filtered_pl = df_pl.filter(
                        pl.col('index').is_in(common_indices.tolist())
                    ).select(['index', 'lon', 'lat', 'z', 'q', 'blh', 'mass'])
                    
                    result[idx + 1] = filtered_pl.to_pandas().set_index('index')
                else:
                    result[idx + 1] = pd.DataFrame(columns=['lon', 'lat', 'z', 'q', 'blh', 'mass'])
                
                print('.', end='')
            except Exception as e:

                print(f"\n! Process file {f} Failure: {e}")
                result[idx + 1] = pd.DataFrame(columns=['lon', 'lat', 'z', 'q', 'blh', 'mass'])
        
        output_path = f"{config.TEMPORARY_FILE_PATH}/{times}-{config.TIME_SPAN}_P_particle_backward_files.pkl"
        writepkl(output_path, result)
        print("\nTracking files complete")
        
        print("Start to establish particle trajectory table")
        
        df0 = [None] * len(particle_indices)
        
        track_times = pd.to_datetime(times, format='%Y%m%d%H') - pd.Timedelta(hours=config.TIME_SPAN) * np.arange(len(result) - 1)
        
        if df_method == 'on':
            try:
                df_BLH_factor = get_df_BLH_factor(config.DF_FILE_PATH, times, config.TRACKING_DAYS, direction='backward')
            except Exception as e:
                print(f"! Failed to get BLH factor: {e}")        
        
        batch_size = 200
        for batch_start in range(0, len(particle_indices), batch_size):
            batch_end = min(batch_start + batch_size, len(particle_indices))
            batch_indices = particle_indices[batch_start:batch_end]
            
            for idx, index in enumerate(batch_indices):
                global_idx = batch_start + idx
                try:
                    particle_data = []
                    
                    particle_data.append(result[0].loc[index, ['lon', 'lat', 'q_diff', 'mass', 'p_mass']])
                    
                    for j in range(1, len(result)):
                        if index in result[j].index:
                            particle_data.append(result[j].loc[index, ['lon', 'lat', 'z', 'q', 'blh']])
                        else:
                            particle_data.append(pd.Series({'lon': np.nan, 'lat': np.nan, 'z': np.nan, 'q': np.nan, 'blh': np.nan}))
                    
                    df_index = pd.DataFrame(particle_data).copy()
                    
                    if len(df_index) > 2:
                        df_pl = pl.from_pandas(df_index)
                        
                        lat_vals = df_pl.select('lat').to_numpy().flatten()
                        lon_vals = df_pl.select('lon').to_numpy().flatten()
                        z_vals = df_pl.select('z').to_numpy().flatten()
                        q_vals = df_pl.select('q').to_numpy().flatten()
                        blh_vals = df_pl.select('blh').to_numpy().flatten()
                        
                        new_lat = lat_vals.copy()
                        new_lon = lon_vals.copy()
                        new_z = z_vals.copy()
                        new_q_diff = np.zeros(len(df_index))
                        new_q_diff[0] = df_index['q_diff'].values[0]
                        new_blh = blh_vals.copy()
                        
                        new_lat[1:-1] = (lat_vals[1:-1] + lat_vals[2:]) / 2
                        new_lon[1:-1] = (lon_vals[1:-1] + lon_vals[2:]) / 2
                        new_z[1:-1] = (z_vals[1:-1] + z_vals[2:]) / 2
                        new_q_diff[1:-1] = q_vals[1:-1] - q_vals[2:]
                        new_blh[1:-1] = (blh_vals[1:-1] + blh_vals[2:]) / 2
                        
                        df_index = df_index.assign(lat=new_lat,
                                                   lon=new_lon,
                                                   z=new_z,
                                                   q_diff=new_q_diff,
                                                   blh=new_blh)
                    
                    df_index = df_index.iloc[:-1]
                    
                    if df_method == 'on':
                        df_index['lat_round'] = calculate_coordinate_round(df_index['lat'], config.OUTPUT_SPATIAL_RESOLUTION)
                        df_index['lon_round'] = calculate_coordinate_round(df_index['lon'], config.OUTPUT_SPATIAL_RESOLUTION)
                        df_index['start_time'] = track_times
                        
                        try:

                            df_index_pl = pl.from_pandas(df_index.reset_index(drop=True))
                            df_BLH_factor_pl = pl.from_pandas(df_BLH_factor.reset_index(drop=True))
                            
                            df_index_pl = df_index_pl.with_columns([
                                pl.col("lat_round").cast(pl.Float64),
                                pl.col("lon_round").cast(pl.Float64)
                            ])
                            df_BLH_factor_pl = df_BLH_factor_pl.with_columns([
                                pl.col("latitude").cast(pl.Float64),
                                pl.col("longitude").cast(pl.Float64)
                            ])
                            
                            merged_pl = df_index_pl.join(
                                df_BLH_factor_pl,
                                left_on=['lat_round', 'lon_round', 'start_time'],
                                right_on=['latitude', 'longitude', 'time'],
                                how='left'
                            )
                            
                            df_index = merged_pl.to_pandas()
                            
                            drop_cols = ['latitude', 'longitude', 'start_time', 'time', 'lat_round', 'lon_round']
                            df_index = df_index.drop(columns=[col for col in drop_cols if col in df_index.columns])
                            
                            mask = (df_index['q_diff'] > params['q_diff_e']) & (df_index['z'] < (df_index['BLH_factor'] * df_index['blh']))
                            df_index['f0'] = np.where(mask, df_index['q_diff'] / df_index['q'], 0)
                            
                            if 'BLH_factor' in df_index.columns:
                                df_index = df_index.drop(columns=['BLH_factor'])
                                
                        except Exception as e:
                            print(f"! Error in DF BLH_factor processing: {e}")

                    else:
                        mask = (df_index['q_diff'] > params['q_diff_e']) & (df_index['z'] < (params['blh_factor'] * df_index['blh']))
                        df_index['f0'] = np.where(mask, df_index['q_diff'] / df_index['q'], 0)
                    
                    df_index['f'] = df_index['f0'].copy()
                    for l in range(2, len(df_index)):
                        df_index.iloc[l:, df_index.columns.get_loc('f')] = df_index['f'].iloc[l:] * (1 - df_index['f0'].iloc[l - 1])
                                               
                    df_index = df_index.drop(columns=['f0'])
                    
                    df0[global_idx] = df_index
                except Exception as e:
                    print(f"! Particle tracking {index} Failure: {e}")
            
            print(f"Build tracking table for {times} {batch_end}/{len(particle_indices)}")
        
        output_path = os.path.join(config.TEMPORARY_FILE_PATH,
                                   f"{times}-{config.TIME_SPAN}_P_particle_tracking_table.pkl")
        writepkl(output_path, df0)
        
        del result
        gc.collect()
        
        print(f"Tracking table for {times} has been established")
        return df0
    #%% ==========================================================================
    
    for i in range(1, len(files_list)):
        try:
            get_and_track_particles(files_list[i], files_list[i - 1], df_method, i)
        except Exception as e:
            print(f'! get_and_track_particles generated an exception: {e}')

    end_time = time.time()
    execution_time = end_time - begin_time
    print(f"Step 3 done, total {execution_time} seconds")


if __name__ == "__main__":
    filtering_and_tracking('WaterSip-DF-HAMSTER', 'on') 
