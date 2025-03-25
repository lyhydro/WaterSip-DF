# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
import xarray as xr
from main_functions import get_files, readpartposit_to_df, from_shp_get_parts, from_bounds_get_parts, calculate_RH, \
    midpoint, back_tracking_files, get_df_BLH_factor, writepkl, calculate_coordinate_round
import time
from YAMLConfig import YAMLConfig
import logging
import gc

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class FilteringAndTracking:

    def __init__(self, WaterSip_method):
        config = YAMLConfig('config.yaml')
        # General setting
        general_config = config.get('General')
        self.WaterSip_method = WaterSip_method 
        self.tracking_days = general_config['tracking_days']
        self.time_span = general_config['time_span']
        self.output_spatial_resolution = general_config['output_spatial_resolution']
        self.start_time = str(general_config['start_time'])
        self.end_time = str(general_config['end_time'])
        self.partposit_path = general_config['partposit_path']
        self.target_region = general_config['target_region']
        self.temporary_file_path = general_config['temporary_file_path']
        # default WaterSip setting
        self.default_q_diff_p = 0
        self.default_q_diff_e = 0.0002
        self.default_rh_threshold = 80
        self.default_blh_factor = 1.5
        #
        if WaterSip_method=='WaterSip':
            self.q_diff_p = config.get('WaterSip')['q_diff_p']
            self.q_diff_e = config.get('WaterSip')['q_diff_e']
            self.rh_threshold = config.get('WaterSip')['rh_threshold']
            self.blh_factor = config.get('WaterSip')['blh_factor']
        elif WaterSip_method=='WaterSip-HAMSTER':
            self.q_diff_p = config.get('WaterSip-HAMSTER')['q_diff_p']
            self.q_diff_e = config.get('WaterSip-HAMSTER')['q_diff_e']
            self.rh_threshold = config.get('WaterSip-HAMSTER')['rh_threshold']
            self.blh_factor = config.get('WaterSip-HAMSTER')['blh_factor']
        elif WaterSip_method=='WaterSip-DF':
            self.q_diff_p = config.get('WaterSip-DF')['q_diff_p_df']
            self.q_diff_e = config.get('WaterSip-DF')['q_diff_e_df']
            self.DF_file_path = config.get('WaterSip-DF')['DF_file_path'] 
        elif WaterSip_method=='WaterSip-DF-HAMSTER':
            self.q_diff_p = config.get('WaterSip-DF-HAMSTER')['q_diff_p_df']
            self.q_diff_e = config.get('WaterSip-DF-HAMSTER')['q_diff_e_df']
            self.DF_file_path = config.get('WaterSip-DF-HAMSTER')['DF_file_path'] 
        #
        self.temp_rh_threshold = {}
        # 
        self.file_cache = {}
        # 
        self.max_cache_size = general_config['max_cache_size']
        # 
        self.blh_factor_cache = {}
        # 
        self.files_list = None
        # 
        self.tracking_results_cache = {}

    # %% * * * * * * * * * * * * * * * * * * * * * MAIN - found all P particles * * * * * * * * * * * * * * * * * * * * *
    def filtering_and_tracking(self, df_method):
        begin_time = time.time()
        print(f'Step 3 ({self.WaterSip_method}): Particle filtering and moisture tracking...')
        print(f"start_time：{self.start_time}")
        print(f"end_time：{self.end_time}")
        print(f"tracking_days：{self.tracking_days}")
        # 
        self.files_list = get_files(self.partposit_path, self.start_time, self.end_time, self.time_span)      
        # 
        self._preload_files(self.files_list[:min(self.max_cache_size, len(self.files_list))])   
        # 
        for i in range(1, len(self.files_list)):
            try:
                self.calculate_q_diff(self.files_list[i], self.files_list[i - 1], df_method, i)
            except Exception as e:
                print(f'!!! calculate_q_diff generated an exception: {e}')
                logging.error(f"!!! Error processing file {self.files_list[i]}: {e}", exc_info=True)

        end_time = time.time()
        execution_time = end_time - begin_time
        print(f"\nProgram execution time: {execution_time} seconds")
        print('Step 3 done!')
        # %%

    def _preload_files(self, files):
        print(f"Preload {len(files)} start files into cache")
        for f in files:
            if f not in self.file_cache:
                try:
                    df = readpartposit_to_df(f, variables=['lon', 'lat', 'z', 'q', 'blh', 'mass', 't', 'dens'])
                    self.file_cache[f] = df
                except Exception as e:
                    print(f"!!! Preloading file {f} failed: {e}")
        print("Preloading complete.")

    def calculate_q_diff(self, file, prvfile, df_method, i):
        # Record the start time of the function
        start_time = time.time()
        print(f"\nStrat identify P particles for {os.path.basename(prvfile)[-14:-4]}")
        
        # Read from cache or load file from disk
        if file in self.file_cache:
            df = self.file_cache[file]
        else:
            df = readpartposit_to_df(file, variables=['lon', 'lat', 'z', 'q', 'blh', 'mass', 't', 'dens'])
            self.file_cache[file] = df
            
        if prvfile in self.file_cache:
            df0 = self.file_cache[prvfile]
        else:
            df0 = readpartposit_to_df(prvfile, variables=['lon', 'lat', 'z', 'q', 'blh', 't', 'dens'])
            self.file_cache[prvfile] = df0
        
        # 
        if 'RH' not in df.columns:
            calculate_RH(df)
        if 'RH' not in df0.columns:
            calculate_RH(df0, RH_name='RH', dens='dens', q='q', t='t')
    
        # Use a more efficient merging method.
        common_indices = df.index.intersection(df0.index)        
        # 
        df = df.loc[common_indices]
        df0 = df0.loc[common_indices]        

        # 
        result_df = pd.DataFrame({
            'lat': df['lat'].values,
            'lon': df['lon'].values,
            'lat0': df0['lat'].values,
            'lon0': df0['lon'].values,
            'q': df['q'].values,
            'q0': df0['q'].values,
            'RH': df['RH'].values,
            'RH0': df0['RH'].values,
            'mass': df['mass'].values
        }, index=common_indices)
        
        # q_diff
        result_df['q_diff'] =  result_df['q'] - result_df['q0']        
        result_df = result_df.drop(columns=['q','q0'])
               
        # Filtering q_diff
        result_df = result_df[ result_df['q_diff'] < self.q_diff_p ]
        
        # Calculate the midpoint coordinates
        result_df['lat'], result_df['lon'] = midpoint(result_df['lat'].values, 
                                                      result_df['lon'].values, 
                                                      result_df['lat0'].values, 
                                                      result_df['lon0'].values )       
        result_df = result_df.drop(columns=['lat0', 'lon0'])
        
        # Filtering particles in the research area
        if self.target_region[-4:] == '.shp':
            result_df = from_shp_get_parts(result_df, self.target_region)
        elif isinstance(self.target_region, list):
            result_df = from_bounds_get_parts(result_df, 
                                              self.target_region[0], 
                                              self.target_region[1], 
                                              self.target_region[2],
                                              self.target_region[3] )
        
        # Calculate the mid RH
        result_df['RH'] = (result_df['RH'] + result_df['RH0']) / 2
        result_df = result_df.drop(columns=['RH0'])
        
        # Filter according to the method
        tmp_time = prvfile[-14:-4]
        if df_method == 'on':
            # Calculate the coordinate grid
            result_df['lat_round'] = calculate_coordinate_round(result_df['lat'], self.output_spatial_resolution)
            result_df['lon_round'] = calculate_coordinate_round(result_df['lon'], self.output_spatial_resolution)
            
            # get DF_RH_thresholds and transfor to dataframe
            if tmp_time not in self.temp_rh_threshold:
                try:
                    rh_file = f'{self.DF_file_path}/DF_RH_thresholds_{tmp_time}.nc'
                    RH_threshold = xr.open_dataset(rh_file)['DF_RH_thresholds']
                    self.temp_rh_threshold[tmp_time] = RH_threshold
                except Exception as e:
                    print(f"!!! Failed to read the RH threshold file: {e}")
                    # Using default values
                    result_df = result_df[result_df['q_diff'] < self.default_q_diff_p]
                    result_df = result_df[result_df['RH'] > self.default_rh_threshold]
                    result_df = result_df.drop(columns=['RH'])
            else:
                RH_threshold = self.temp_rh_threshold[tmp_time]
            
            # Convert to DataFrame and filter
            try:
                df_RH_threshold = RH_threshold.to_dataframe(name='RH_threshold').reset_index()
                df_RH_threshold = df_RH_threshold.query("RH_threshold != 100 and RH_threshold.notna()", engine='numexpr')
                
                # merging 
                original_index = result_df.index  # keep index
                result_df = pd.merge(result_df, 
                                     df_RH_threshold, 
                                     how='left', 
                                     left_on=['lat_round', 'lon_round'],
                                     right_on=['latitude', 'longitude'] )
                result_df.index = original_index  # recover index
                
                # Filter RH
                result_df = result_df[ result_df['RH'] >= result_df['RH_threshold'] ]
                
                # Delete unnecessary columns
                result_df = result_df.drop(columns=['latitude', 'longitude', 'RH', 
                                                    'RH_threshold', 'lat_round', 'lon_round' ])
            
            except Exception as e:
                print(f"!!! RH threshold processing failed: {e}")
                # Using default values
                result_df = result_df[result_df['q_diff'] < self.default_q_diff_p]
                result_df = result_df[result_df['RH'] > self.default_rh_threshold]
                result_df = result_df.drop(columns=['RH', 'lat_round', 'lon_round'])
        else:
            # Use WaterSip
            result_df = result_df[result_df['q_diff'] < self.q_diff_p]
            result_df = result_df[result_df['RH'] > self.rh_threshold]
            result_df = result_df.drop(columns=['RH'])
        
        # Calculate p_mass
        result_df['p_mass'] = -result_df['mass'] * result_df['q_diff']
        
        # Track particles
        self.track_particles(i - 1, tmp_time, result_df, df_method)
        
        # Record processing time
        end_time = time.time()
        print(f"\n{os.path.basename(file)[-14:-4]} processing complete, total duration: {end_time - start_time:.2f} seconds")
        
        # Clear Memory
        del df, df0, result_df
        if len(self.file_cache) > self.max_cache_size:
            # Delete the earliest cache item
            oldest_key = next(iter(self.file_cache))
            del self.file_cache[oldest_key]
        gc.collect()

    def track_particles(self, i, times, all_P_parts, df_method):
        # Check if it has been processed already
        cache_key = f"{times}_{df_method}"
        if cache_key in self.tracking_results_cache:
            print(f"Using cached tracking results: {times}")
            return self.tracking_results_cache[cache_key]
            
        # Record the start time
        track_start = time.time()
        print(f"Start track particles for {times}, particle number: {len(all_P_parts)}")
        
        if len(all_P_parts) == 0:
            print(f"!!! Warning: {times} No particles found")
            return
            
        # Get the list of backtrack files
        files_track = back_tracking_files(self.partposit_path, times, self.tracking_days, self.time_span)

        # Preload files
        for f in files_track:
            if f not in self.file_cache and len(self.file_cache) < self.max_cache_size:
                try:
                    # Ensure that all the required columns are read.
                    df = readpartposit_to_df(f, variables=['lon', 'lat', 'z', 'q', 'blh', 'mass'])
                    self.file_cache[f] = df
                except Exception as e:
                    print(f"!!! Failed to load file {f}: {e}")
        
        # Initialize a result list with a pre-allocated size.
        result = [None] * (len(files_track) + 1)
        result[0] = all_P_parts
        
        # Obtain initial particle indices
        particle_indices = all_P_parts.index.values
        
        # Process each backtrack file
        for idx, f in enumerate(files_track):
            try:
                if f in self.file_cache:
                    df = self.file_cache[f]
                else:
                    try:
                        df = readpartposit_to_df(f, variables=['lon', 'lat', 'z', 'q', 'blh', 'mass'])
                        # If the cache is not full, add to the cache.
                        if len(self.file_cache) < self.max_cache_size:
                            self.file_cache[f] = df
                    except Exception as e:
                        print(f"\n!!! Failed to read file {f}: {e}")
                        # Create an empty DataFrame with all the necessary columns.
                        df = pd.DataFrame(columns=['lon', 'lat', 'z', 'q', 'blh', 'mass'])
                
                # Ensure that the DataFrame contains all the required columns.
                required_columns = ['lon', 'lat', 'z', 'q', 'blh', 'mass']
                for col in required_columns:
                    if col not in df.columns:
                        print(f"!!! Adding missing column {col} to DataFrame")
                        df[col] = np.nan
                
                # Find the common particles
                common_indices = np.intersect1d(particle_indices, df.index.values)
                
                # Update particle index
                particle_indices = common_indices
                
                # Select only the required columns and index
                if len(common_indices) > 0:
                    result[idx + 1] = df.loc[common_indices, required_columns].copy()
                else:
                    # If there is no common index, create an empty DataFrame.
                    result[idx + 1] = pd.DataFrame(columns=required_columns)
                
                print('.', end='')
            except Exception as e:
                print(f"\n!!! Process file {f} Failure: {e}")
                # Using an empty DataFrame, ensure that it includes all the necessary columns
                result[idx + 1] = pd.DataFrame(columns=['lon', 'lat', 'z', 'q', 'blh', 'mass'])
        
        # Save backtracking file results
        output_path = f"{self.temporary_file_path}/{times}-{self.time_span}_P_particle_backward_files.pkl"
        writepkl(output_path, result)
        print(f"\nTracking process complete, total duration: {time.time() - track_start:.2f} seconds")
        
        # Processing particle trajectory table
        print("Start to establish particle trajectory table")
        
        # Obtain all particle indices
        indexs = result[0].index.values
        
        # Pre-allocate result list
        df0 = [None] * len(indexs)
        
        # Precompute time series
        track_times = pd.to_datetime(times, format='%Y%m%d%H') - pd.Timedelta(hours=self.time_span) * np.arange(len(result) - 1)
        
        # Preloading BLH factors
        if df_method == 'on':
            blh_key = f"{times}_{self.tracking_days}_backward"
            if blh_key not in self.blh_factor_cache:
                try:
                    self.blh_factor_cache[blh_key] = get_df_BLH_factor(self.DF_file_path, times, self.tracking_days, direction='backward')
                except Exception as e:
                    print(f"!!! Failed to load BLH factor: {e}")
        
        # Bulk processing of particles
        batch_size = 100  # Number of particles processed per batch
        for batch_start in range(0, len(indexs), batch_size):
            batch_end = min(batch_start + batch_size, len(indexs))
            batch_indices = indexs[batch_start:batch_end]
            
            # Handling particles in batches
            for idx, index in enumerate(batch_indices):
                global_idx = batch_start + idx
                try:
                    # Collect particle data
                    particle_data = []
                    
                    # Add initial point data
                    first_point_columns = ['lon', 'lat', 'q_diff', 'mass', 'p_mass']
                    # Ensure that all columns exist
                    if not all(col in result[0].columns for col in first_point_columns):
                        missing_cols = [col for col in first_point_columns if col not in result[0].columns]
                        print(f"!!! Missing columns in result[0]: {missing_cols}")
                        # Add default values for missing columns
                        for col in missing_cols:
                            result[0][col] = np.nan
                    
                    particle_data.append(result[0].loc[index, first_point_columns])
                    
                    # Collect particle data for all time steps
                    for j in range(1, len(result)):
                        if j < len(result) and result[j] is not None and index in result[j].index:
                            # Ensure that all required columns are present
                            required_cols = ['lon', 'lat', 'z', 'q', 'blh']
                            if not all(col in result[j].columns for col in required_cols):
                                missing_cols = [col for col in required_cols if col not in result[j].columns]
                                print(f"!!! Missing columns in result[{j}]: {missing_cols}")
                                # Add default values for missing columns
                                for col in missing_cols:
                                    result[j][col] = np.nan
                            
                            particle_data.append(result[j].loc[index, required_cols])
                        else:
                            # Using NaN values
                            particle_data.append(pd.Series({'lon': np.nan, 'lat': np.nan, 'z': np.nan, 'q': np.nan, 'blh': np.nan}))
                    
                    # create DataFrame
                    df_index = pd.DataFrame(particle_data).copy()
                    
                    # Ensure that all required columns are present
                    required_cols = ['lon', 'lat', 'z', 'q', 'blh', 'q_diff', 'mass', 'p_mass']
                    for col in required_cols:
                        if col not in df_index.columns:
                            df_index[col] = np.nan
                    
                    # Calculate the midpoint (using vectorized operations)
                    if len(df_index) > 2:
                        # 获取数组
                        lat_vals = df_index['lat'].values
                        lon_vals = df_index['lon'].values
                        z_vals = df_index['z'].values
                        q_vals = df_index['q'].values
                        blh_vals = df_index['blh'].values
                        
                        # Create a new array for midpoint calculation
                        new_lat = lat_vals.copy()
                        new_lon = lon_vals.copy()
                        new_z = z_vals.copy()
                        new_q_diff = np.zeros(len(df_index))
                        new_q_diff[0] = df_index['q_diff'].values[0]
                        new_blh = blh_vals.copy()
                        
                        # Calculate the midpoint
                        new_lat[1:-1] = (lat_vals[1:-1] + lat_vals[2:]) / 2
                        new_lon[1:-1] = (lon_vals[1:-1] + lon_vals[2:]) / 2
                        new_z[1:-1] = (z_vals[1:-1] + z_vals[2:]) / 2
                        new_q_diff[1:-1] = q_vals[1:-1] - q_vals[2:]
                        new_blh[1:-1] = (blh_vals[1:-1] + blh_vals[2:]) / 2  # could change to max-blh between two steps, do same change in step 2 
                        
                        # 一次性更新DataFrame，避免链式索引
                        df_index = df_index.assign(lat=new_lat,
                                                   lon=new_lon,
                                                   z=new_z,
                                                   q_diff=new_q_diff,
                                                   blh=new_blh )
                    
                    # Remove the last line
                    df_index = df_index.iloc[:-1]
                    
                    # Handling the DF method
                    if df_method == 'on':
                        # Calculate the coordinate grid
                        df_index['lat_round'] = calculate_coordinate_round(df_index['lat'], self.output_spatial_resolution)
                        df_index['lon_round'] = calculate_coordinate_round(df_index['lon'], self.output_spatial_resolution)
                        df_index['start_time'] = track_times
                        
                        # Obtain the BLH factor
                        blh_key = f"{times}_{self.tracking_days}_backward"
                        if blh_key in self.blh_factor_cache:
                            df_BLH_factor = self.blh_factor_cache[blh_key]
                        else:
                            try:
                                df_BLH_factor = get_df_BLH_factor(self.DF_file_path, times, self.tracking_days, direction='backward')
                                self.blh_factor_cache[blh_key] = df_BLH_factor
                            except Exception as e:
                                print(f"!!! Failed to get BLH factor: {e}")
                                # Create a default BLH factor
                                df_BLH_factor = pd.DataFrame(columns=['latitude', 'longitude', 'time', 'BLH_factor'])
                        
                        try:
                            # Merge data
                            df_index = pd.merge( df_index, df_BLH_factor, how='left', 
                                                left_on=['lat_round', 'lon_round', 'start_time'], 
                                                right_on=['latitude', 'longitude', 'time'] )
                            
                            # Delete unnecessary columns
                            drop_cols = ['latitude', 'longitude', 'start_time', 'time', 'lat_round', 'lon_round']
                            df_index = df_index.drop(columns=[col for col in drop_cols if col in df_index.columns])
                            
                            # If the "BLH_factor" column does not exist, add a default value.
                            if 'BLH_factor' not in df_index.columns:
                                df_index['BLH_factor'] = self.default_blh_factor
                            
                            # Calculate f0 (using vectorized operations)
                            mask = (df_index['q_diff'] > self.q_diff_e) & (df_index['z'] < (df_index['BLH_factor'] * df_index['blh']))
                            df_index['f0'] = np.where(mask, df_index['q_diff'] / df_index['q'], 0)
                            
                            # Delete the "BLH_factor" column
                            if 'BLH_factor' in df_index.columns:
                                df_index = df_index.drop(columns=['BLH_factor'])
                        except Exception as e:
                            print(f"!!! Error in DF method processing: {e}")
                            # Use the default method
                            mask = (df_index['q_diff'] > self.default_q_diff_e) & (df_index['z'] < (self.default_blh_factor * df_index['blh']))
                            df_index['f0'] = np.where(mask, df_index['q_diff'] / df_index['q'], 0)
                    else:
                        # Use WaterSip
                        mask = (df_index['q_diff'] > self.q_diff_e) & (df_index['z'] < (self.blh_factor * df_index['blh']))
                        df_index['f0'] = np.where(mask, df_index['q_diff'] / df_index['q'], 0)
                    
                    # # Delete unnecessary columns
                    # drop_cols = ['z', 'blh', 'q_diff', 'q']
                    # df_index = df_index.drop(columns=[col for col in drop_cols if col in df_index.columns])
                    
                    # Calculate the f-value (using vectorized operations)
                    df_index['f'] = df_index['f0'].copy()
                    # if len(df_index) > 1:
                    #     shifted_f0 = df_index['f0'].shift(1)
                    #     mask = ~pd.isna(shifted_f0)
                    #     df_index.loc[mask, 'f'] = df_index.loc[mask, 'f'] * (1 - shifted_f0[mask])
                    for l in range(2, len(df_index)):
                        df_index.iloc[l:, df_index.columns.get_loc('f')] = df_index['f'].iloc[l:] * (1 - df_index['f0'].iloc[l - 1])
                                               
                    # Delete the "f0" column
                    if 'f0' in df_index.columns:
                        df_index = df_index.drop(columns=['f0'])
                    
                    # Store the results
                    df0[global_idx] = df_index
                except Exception as e:
                    print(f"!!! Particle Processing {index} Failure: {e}")
                    # Using an empty DataFrame
                    df0[global_idx] = pd.DataFrame()
            
            # Display progress
            print(f"Build tracking table for {times} {batch_end}/{len(indexs)}")
        
        # Save the result
        output_path = os.path.join(self.temporary_file_path,
                                   f"{times}-{self.time_span}_P_particle_tracking_table.pkl" )
        writepkl(output_path, df0)
        
        # Cache the result
        self.tracking_results_cache[cache_key] = df0
        
        # Clear Memory
        del result
        gc.collect()
        
        print(f"\nTracking table for {times} has been established, with total duration: {time.time() - track_start:.2f} seconds")
        return df0

if __name__ == "__main__":
    ft = FilteringAndTracking('WaterSip-DF-HAMSTER')
    ft.filtering_and_tracking('on')
