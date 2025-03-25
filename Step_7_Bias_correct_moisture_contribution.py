# -*- coding: utf-8 -*-
import numpy as np
import xarray as xr
import pandas as pd
from main_functions import get_and_combine_obs_files, global_gridcell_info, from_shp_get_mask, from_bounds_get_mask, \
    write_to_nc_2d
from YAMLConfig import YAMLConfig
import logging
import psutil
import gc
from concurrent.futures import ThreadPoolExecutor
import threading
import os
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BiasCorrectMoistureContribution:

    def __init__(self, WaterSip_method):
        config = YAMLConfig('config.yaml')
        general_config = config.get('General')
        self.WaterSip_method = WaterSip_method 
        self.tracking_days = general_config['tracking_days']
        self.start_time = str(general_config['start_time'])
        self.end_time = str(general_config['end_time'])
        self.final_output_path = general_config['final_output_path']
        self.target_region = general_config['target_region']
        # 
        self.data_cache = {}
        self.cache_lock = threading.Lock()
        self.memory_threshold = 75  
        self.max_cache_size = general_config['max_cache_size'] 
        self.max_workers = general_config['max_workers']
        #
        if WaterSip_method=='WaterSip-HAMSTER':
            self.observation_path = config.get('WaterSip-HAMSTER')['observation_path']
            self.P_E_simulation_output_path = config.get('WaterSip-HAMSTER')['P_E_simulation_output_path']
        elif WaterSip_method=='WaterSip-DF-HAMSTER':        
            self.observation_path = config.get('WaterSip-DF-HAMSTER')['observation_path']
            self.P_E_simulation_output_path = config.get('WaterSip-DF-HAMSTER')['P_E_simulation_output_path']


    def clear_memory(self):
        with self.cache_lock:
            self.data_cache.clear()
        gc.collect()
        if psutil.virtual_memory().percent > self.memory_threshold:
            time.sleep(2)  

    def load_cached_data(self, file_path, loader_func, *args):
        with self.cache_lock:
            if file_path in self.data_cache:
                return self.data_cache[file_path]
            
            if len(self.data_cache) >= self.max_cache_size:
                oldest_key = next(iter(self.data_cache))
                del self.data_cache[oldest_key]
                gc.collect()
            
            data = loader_func(file_path, *args)
            self.data_cache[file_path] = data
            return data

    def process_time_chunk(self, time_chunk, obs_type='e'):
        try:
            ds = get_and_combine_obs_files(self.observation_path, time_chunk.strftime('%Y%m').drop_duplicates(), 
                                         variable=obs_type)
            if obs_type == 'e':
                obs_data = -ds.resample(time='6H').sum(dim='time') * 1000
            else:
                obs_data = ds.resample(time='6H').sum(dim='time') * 1000
            
            obs_data = obs_data.sel(time=time_chunk)
            obs_data = obs_data.where(obs_data > 0, 0)
            return obs_data
        except Exception as e:
            logger.error(f"Error processing time chunk: {e}")
            return None

    def process_simulation_data(self, time_point, sim_type='E'):
        try:
            file_path = f'{self.P_E_simulation_output_path}/{sim_type}_simulation_mm_{time_point.strftime("%Y%m%d%H")}.nc'
            data = xr.open_dataset(file_path)[f'{sim_type}_simulation_mm']
            return data.where(data > 0, 0)
        except Exception as e:
            logger.error(f"Error processing simulation data for {time_point}: {e}")
            return None

    def load_moisture_contribution(self, df_method):
        try:
            mc_file = f'{self.final_output_path}/moisture_contribution_mm_{"DF_" if df_method == "on" else ""}{self.start_time}_{self.end_time}.nc'
            
            # 
            if not os.path.exists(mc_file):
                raise FileNotFoundError(f"Moisture contribution file not found: {mc_file}")
            
            # 
            if not os.access(mc_file, os.R_OK):
                raise PermissionError(f"Cannot access moisture contribution file: {mc_file}")
            
            # 
            file_size = os.path.getsize(mc_file)
            if file_size == 0:
                raise ValueError(f"Moisture contribution file is empty: {mc_file}")
            
            logger.info(f"Loading moisture contribution file: {mc_file}")
            
            # 
            with xr.open_dataset(mc_file) as ds:
                if 'moisture_contribution_mm' not in ds:
                    raise KeyError(f"Variable 'moisture_contribution_mm' not found in file: {mc_file}")
                
                # 
                data = ds['moisture_contribution_mm']
                if data.size == 0:
                    raise ValueError(f"No valid data found in moisture contribution file: {mc_file}")
                
                return data
            
        except (OSError, IOError) as e:
            logger.error(f"IO Error reading moisture contribution file: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error reading moisture contribution file: {e}")
            raise

    # %% * * * * * * * * * * MAIN * * * * * * * * * *
    def bias_correct_moisture_contribution(self, df_method):
        try:
            # 
            moisture_contribution_mm = self.load_moisture_contribution(df_method)

            time_p = pd.to_datetime(moisture_contribution_mm['time'].values)
            time_e = pd.date_range(time_p[0] - pd.Timedelta(days=self.tracking_days), time_p[-1], freq='6h')

            # 
            latitude, longitude, gridcell_area = global_gridcell_info(1, lat_nor=90, lat_sou=-90, lon_lef=-179, lon_rig=180)
            if self.target_region[-4:] == '.shp':
                mask = from_shp_get_mask(self.target_region, latitude, longitude)
            else:
                mask = from_bounds_get_mask(*self.target_region, latitude, longitude)

            chunk_size = 50 

            # 
            logger.info("Processing evaporation data...")
            obs_e_chunks = []
            simu_e_chunks = []
            
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # 
                for i in range(0, len(time_e), chunk_size):
                    chunk = time_e[i:i + chunk_size]
                    obs_e_chunks.append(executor.submit(self.process_time_chunk, chunk, 'e'))
                
                # 
                for t in time_e:
                    simu_e_chunks.append(executor.submit(self.process_simulation_data, t, 'E'))

            # 
            obs_e = xr.concat([f.result() for f in obs_e_chunks if f.result() is not None], dim="time")
            simu_global_e = xr.concat([f.result() for f in simu_e_chunks if f.result() is not None], dim="time")

            # 
            cc_e = obs_e.sum(dim='time') / simu_global_e.sum(dim='time')
            cc_e = cc_e.where(cc_e <= 10, 10).where(cc_e > 0, 0)
            logger.info('Evaporation correction coefficient calculated')

            # 
            logger.info("Processing precipitation data...")
            obs_p_chunks = []
            simu_p_chunks = []
            
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                for i in range(0, len(time_p), chunk_size):
                    chunk = time_p[i:i + chunk_size]
                    obs_p_chunks.append(executor.submit(self.process_time_chunk, chunk, 'tp'))
                
                for t in time_p:
                    simu_p_chunks.append(executor.submit(self.process_simulation_data, t, 'P'))

            obs_p = xr.concat([f.result() for f in obs_p_chunks if f.result() is not None], dim="time")
            simu_global_p = xr.concat([f.result() for f in simu_p_chunks if f.result() is not None], dim="time")

            # 
            moisture_contribution_mm_sum = moisture_contribution_mm.sum(dim='time')
            cc_p = np.sum(obs_p * gridcell_area * mask) * (np.sum(moisture_contribution_mm_sum * gridcell_area) / 
                   np.sum(simu_global_p * gridcell_area * mask)) / np.sum(moisture_contribution_mm_sum * gridcell_area * cc_e)
            
            logger.info(f'Precipitation correction coefficient: {cc_p.values}')

            # 
            moisture_contribution_mm_sum_corrected = moisture_contribution_mm_sum * cc_e * cc_p
            
            # 
            output_name = f'moisture_contribution_mm_{"DF_HAMSTER" if df_method == "on" else "HAMSTER"}_{self.start_time}_{self.end_time}'
            write_to_nc_2d(moisture_contribution_mm_sum_corrected, 'moisture_contribution_mm',
                          output_name, latitude, longitude, self.final_output_path)
            
            logger.info("Bias correction completed successfully")

        except Exception as e:
            logger.error(f"Error in bias correction: {e}")
            raise
        finally:
            self.clear_memory()


if __name__ == "__main__":
    bc = BiasCorrectMoistureContribution()
    bc.bias_correct_moisture_contribution('on')
