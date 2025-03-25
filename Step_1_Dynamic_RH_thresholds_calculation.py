# -*- coding: utf-8 -*-
from collections import deque
import pandas as pd
import numpy as np
from main_functions import get_and_combine_obs_files, get_files, readpartposit_to_df, global_gridcell_info, \
    calculate_RH, midpoint, write_to_nc_2d, calculate_coordinate_round
from concurrent.futures import ThreadPoolExecutor, as_completed
from YAMLConfig import YAMLConfig
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DynamicRHThresholdsCalculation:

    def __init__(self, WaterSip_method):
        config = YAMLConfig('config.yaml')
        general_config = config.get('General')
        self.WaterSip_method = WaterSip_method 
        self.time_span = general_config['time_span']
        self.start_time = str(general_config['start_time'])
        self.end_time = str(general_config['end_time'])
        self.output_spatial_resolution = general_config['output_spatial_resolution']
        self.partposit_path = general_config['partposit_path']
        self.max_workers = general_config['max_workers']
        self.max_cache_size = general_config['max_cache_size']
        # 
        self.temp_files = deque(maxlen=self.max_cache_size)

        if WaterSip_method=='WaterSip-DF':     
            self.q_diff_p = config.get('WaterSip-DF')['q_diff_p_df']        
            self.observation_path = config.get('WaterSip-DF')['observation_path']
            self.DF_file_path = config.get('WaterSip-DF')['DF_file_path']
        elif WaterSip_method=='WaterSip-DF-HAMSTER':     
            self.q_diff_p = config.get('WaterSip-DF-HAMSTER')['q_diff_p_df']        
            self.observation_path = config.get('WaterSip-DF-HAMSTER')['observation_path']
            self.DF_file_path = config.get('WaterSip-DF-HAMSTER')['DF_file_path']

    # %% * * * * * * * * * * * * * * * * * * * * * MAIN * * * * * * * * * * * * * * * * * * * * *
    def dynamic_rh_thresholds_calculation(self):
        logging.info('Step 1 ({self.WaterSip_method}): Calculate dynamic RH thresholds...')

        time = pd.date_range(start=pd.to_datetime(self.start_time, format='%Y%m%d%H'),
                             end=pd.to_datetime(self.end_time, format='%Y%m%d%H'), freq='{}h'.format(self.time_span))[:-1]

        ds = get_and_combine_obs_files(self.observation_path, time.strftime('%Y%m').drop_duplicates(), variable='tp')
        P_era5_6h = ds.resample(time='6H').sum(dim='time') * 1000  # hourly to 6-hourly, m to mm

        files = get_files(self.partposit_path, self.start_time, self.end_time, self.time_span)
        latitude, longitude, gridcell_area = global_gridcell_info(self.output_spatial_resolution, lat_nor=90,
                                                                  lat_sou=-90,
                                                                  lon_lef=-179, lon_rig=180)

        P_era5_6h = P_era5_6h.where(P_era5_6h > 0, 0)  # make p<0 to 0
        P_era5_6h = P_era5_6h.sel(time=time) * gridcell_area  # mm to kg
        DF_RH_thresholds = np.zeros([np.shape(gridcell_area)[0], np.shape(gridcell_area)[1]])

        # 
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_index = {
                executor.submit(self.generate_files, i, files, latitude, longitude, P_era5_6h, time): i
                for i in range(1, len(files)) # from 1 start, the files[0] used to calculate diff
            }
            for future in as_completed(future_to_index):
                i = future_to_index[future]
                try:
                    future.result()
                except Exception as e:
                    logging.error(f'generate_files generated an exception: {e}')

    def generate_files(self, i, files, latitude, longitude, P_era5_6h, time):
        logging.info(f'Processing file {files[i-1][-24:-4]} to {files[i][-14:-4]}')
        # read end-time partposit
        df = self._read_and_cache_file(files[i], variables=['lat', 'lon', 'q', 't', 'dens', 'mass'])
        calculate_RH(df)
        df = df.drop(columns=['t', 'dens'])
        # read one-time-step before partposit
        df0 = self._read_and_cache_file(files[i - 1], variables=['lat', 'lon', 'q', 't', 'dens'])
        df0 = df0.loc[:, ['lon', 'lat', 'q', 't', 'dens']]
        df0 = df0.rename(columns={'lat': 'lat0', 'lon': 'lon0', 'q': 'q0', 't': 't0', 'dens': 'dens0'})
        calculate_RH(df0, RH_name='RH0', dens='dens0', q='q0', t='t0')
        df0 = df0.drop(columns=['t0', 'dens0'])
        #
        df = df.merge(df0, left_index=True, right_index=True) 
        # q_diff filtering
        df['q_diff'] = df['q'] - df['q0']
        df = df[df['q_diff'] < self.q_diff_p]
        df = df.drop(columns=['q', 'q0'])
        # calculate middle-lat and middle-lon
        df['lat'], df['lon'] = midpoint(df['lat'].values, df['lon'].values, df['lat0'].values, df['lon0'].values)
        df = df.drop(columns=['lat0', 'lon0'])
        # mid_RH
        df['RH'] = (df['RH'] + df['RH0']) / 2
        df = df.drop(columns=['RH0'])
        #
        df['lat_round'] = calculate_coordinate_round(df['lat'], self.output_spatial_resolution)
        df['lon_round'] = calculate_coordinate_round(df['lon'], self.output_spatial_resolution)
        df = df.drop(columns=['lon', 'lat'])
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
            lat_idx_p = ((latitude[-1] - df_grouped_p['lat_round']) / self.output_spatial_resolution).astype(int)
            lon_idx_p = ((df_grouped_p['lon_round'] - longitude[0]) / self.output_spatial_resolution).astype(int)
            p_simulation[j, lat_idx_p, lon_idx_p] = df_grouped_p['p_mass']
        obs0 = P_era5_6h[i - 1].values
        min_indices = np.nanargmin(np.abs(p_simulation - obs0), axis=0)
        # 
        DF_RH_thresholds = np.where(np.all(p_simulation == 0, axis=0), np.nan, RH_span[min_indices])
        DF_RH_thresholds[obs0 == 0] = 100  # if no P, RH threshold equals 100
        write_to_nc_2d(DF_RH_thresholds, 'DF_RH_thresholds',
                       'DF_RH_thresholds_{}'.format(time[i - 1].strftime('%Y%m%d%H')), latitude, longitude,
                       self.DF_file_path)
        logging.info(f'{files[i - 1].split("partposit_")[-1][:-4]} done!')
        # 清理内存
        del df, df0, p_simulation, obs0, DF_RH_thresholds

    def _read_and_cache_file(self, file_path, variables):
        for cached_file, cached_data in self.temp_files:
            if cached_file == file_path:
                return cached_data
        # 
        df = readpartposit_to_df(file_path, variables=variables)
        self.temp_files.append((file_path, df))
        return df

if __name__ == "__main__":
    ft = DynamicRHThresholdsCalculation()
    ft.dynamic_rh_thresholds_calculation()