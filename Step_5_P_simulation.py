# -*- coding: utf-8 -*-
from collections import deque
import pandas as pd
import numpy as np
import xarray as xr
from main_functions import get_files, readpartposit_to_df, global_gridcell_info, calculate_RH, midpoint, write_to_nc_2d, \
    calculate_coordinate_round
from YAMLConfig import YAMLConfig
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class PSimulation:

    def __init__(self, WaterSip_method):
        config = YAMLConfig('config.yaml')
        general_config = config.get('General')
        self.WaterSip_method = WaterSip_method 
        self.partposit_path = general_config['partposit_path']
        self.time_span = general_config['time_span']
        self.output_spatial_resolution = general_config['output_spatial_resolution']
        self.start_time = str(general_config['start_time'])
        self.end_time = str(general_config['end_time'])
        self.max_cache_size = general_config['max_cache_size']
        # 
        self.temp_files = deque(maxlen=self.max_cache_size)       
        #
        if WaterSip_method=='WaterSip-HAMSTER':
            self.q_diff_p = config.get('WaterSip-HAMSTER')['q_diff_p']
            self.rh_threshold = config.get('WaterSip-HAMSTER')['rh_threshold']
            self.P_E_simulation_output_path = config.get('WaterSip-HAMSTER')['P_E_simulation_output_path']
        elif WaterSip_method=='WaterSip-DF-HAMSTER':
            self.q_diff_p = config.get('WaterSip-DF-HAMSTER')['q_diff_p_df']
            self.P_E_simulation_output_path = config.get('WaterSip-DF-HAMSTER')['P_E_simulation_output_path']
            self.DF_file_path = config.get('WaterSip-DF-HAMSTER')['DF_file_path']        
        

    # %% * * * * * * * * * * MAIN * * * * * * * * * *
    def p_simulation(self, df_method):
        begin_time = time.time()
        logging.info(f'Step 5 ({self.WaterSip_method}): P simulation...')
        files = get_files(self.partposit_path, self.start_time, self.end_time, self.time_span)
        latitude, longitude, gridcell_area = global_gridcell_info(self.output_spatial_resolution, lat_nor=90,
                                                                  lat_sou=-90,
                                                                  lon_lef=-179, lon_rig=180)
        temp_time = pd.date_range(start=pd.to_datetime(self.start_time, format='%Y%m%d%H'),
                                  end=pd.to_datetime(self.end_time, format='%Y%m%d%H'),
                                  freq='{}h'.format(self.time_span))[:-1]
        P_simulation_mm = np.zeros([np.shape(gridcell_area)[0], np.shape(gridcell_area)[1]])

        for i in range(1, len(files)):  # from 1 start, the files[0] used to calculate diff
            # 
            df_p = self._read_and_cache_file(files[i], variables=['lon', 'lat', 'q', 't', 'dens', 'mass'])
            calculate_RH(df_p)
            df_p = df_p.drop(columns=['dens', 't'])
            # 
            df0 = self._read_and_cache_file(files[i - 1], variables=['lon', 'lat', 'q', 't', 'dens', 'mass'])
            df0 = df0.loc[:, ['lon', 'lat', 'q', 't', 'dens']]
            df0 = df0.rename(columns={'lat': 'lat0', 'lon': 'lon0', 'q': 'q0', 't': 't0', 'dens': 'dens0'})
            calculate_RH(df0, RH_name='RH0', dens='dens0', q='q0', t='t0')
            df0 = df0.drop(columns=['dens0', 't0'])
            #
            df_p = df_p.merge(df0, left_index=True, right_index=True)
            # 
            df_p['q_diff'] = df_p['q'] - df_p['q0']
            df_p = df_p.drop(columns=['q', 'q0'])
            df_p = df_p[df_p['q_diff'] < 0]
            # mid_RH
            df_p['RH'] = (df_p['RH'] + df_p['RH0']) / 2
            df_p = df_p.drop(columns=['RH0'])
            # 
            df_p['lat'], df_p['lon'] = midpoint(df_p['lat'].values, df_p['lon'].values, df_p['lat0'].values,
                                                df_p['lon0'].values)
            df_p = df_p.drop(columns=['lat0', 'lon0'])
            ##
            df_p['lat_round'] = calculate_coordinate_round(df_p['lat'], self.output_spatial_resolution)
            df_p['lon_round'] = calculate_coordinate_round(df_p['lon'], self.output_spatial_resolution)
            df_p = df_p.drop(columns=['lat', 'lon'])
            # df_p = df_p[(df_p['lon_round'] <= 180) & (df_p['lon_round'] >= -179)]
            # 
            if df_method == 'on':
                RH_threshold = \
                    xr.open_dataset(
                        '{}\DF_RH_thresholds_{}.nc'.format(self.DF_file_path, temp_time[i - 1].strftime('%Y%m%d%H')))[
                        'DF_RH_thresholds']
                df_RH_threshold = RH_threshold.to_dataframe(name='RH_threshold').reset_index()
                df_RH_threshold = df_RH_threshold.query("RH_threshold != 100 and RH_threshold.notna()",
                                                        engine='numexpr')
                df_p = pd.merge(df_p, df_RH_threshold, how='left', left_on=['lat_round', 'lon_round'],
                                right_on=['latitude', 'longitude'])
                df_p = df_p.drop(columns=['latitude', 'longitude'])
                df_p.dropna(subset=['RH_threshold'], inplace=True)
                # df_p = df_p[df_p['RH_threshold'] != 100]
                df_p = df_p[df_p['RH'] >= df_p['RH_threshold']]
            else:
                df_p = df_p[df_p['q_diff'] < self.q_diff_p]
                df_p = df_p[df_p['RH'] >= self.rh_threshold]
            # 
            df_p['p_mass'] = -df_p['mass'] * df_p['q_diff']  # kg, *-1 makes P positive
            # 
            df_grouped_p = df_p.groupby(['lat_round', 'lon_round'])['p_mass'].sum().reset_index()
            # 
            lat_idx_p = ((latitude[-1] - df_grouped_p['lat_round']) / self.output_spatial_resolution).astype(int)
            lon_idx_p = ((df_grouped_p['lon_round'] - longitude[0]) / self.output_spatial_resolution).astype(int)
            P_simulation_mm[lat_idx_p, lon_idx_p] = df_grouped_p['p_mass']
            P_simulation_mm = P_simulation_mm / gridcell_area
            write_to_nc_2d(P_simulation_mm, 'P_simulation_mm',
                           'P_simulation_mm_{}'.format(temp_time[i - 1].strftime('%Y%m%d%H')), latitude, longitude,
                           self.P_E_simulation_output_path)
            logging.info(f'{files[i - 1].split("partposit_")[-1][:-4]} P simulation done!')
            # del df_p, df0, P_simulation_mm, df_grouped_p, df_RH_threshold
            del df_p, df0, df_grouped_p

        execution_time = time.time() - begin_time
        logging.info(f"Total duration：{execution_time} seconds")
        logging.info('Step 5 done!')

    def _read_and_cache_file(self, file_path, variables):
        for cached_file, cached_data in self.temp_files:
            if cached_file == file_path:
                return cached_data
        # 
        df = readpartposit_to_df(file_path, variables=variables)
        self.temp_files.append((file_path, df))
        return df


if __name__ == "__main__":
    ft = PSimulation()
    ft.p_simulation('on')
