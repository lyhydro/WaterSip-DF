# -*- coding: utf-8 -*-
from collections import deque
import pandas as pd
import numpy as np
import xarray as xr
from main_functions import get_files, back_tracking_files, readpartposit_to_df, global_gridcell_info, midpoint, \
    write_to_nc_2d, calculate_coordinate_round
from YAMLConfig import YAMLConfig
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class ESimulation:

    def __init__(self, WaterSip_method):
        config = YAMLConfig('config.yaml')
        general_config = config.get('General')
        self.WaterSip_method = WaterSip_method 
        self.partposit_path = general_config['partposit_path']
        self.tracking_days = general_config['tracking_days']
        self.time_span = general_config['time_span']
        self.output_spatial_resolution = general_config['output_spatial_resolution']
        self.start_time = str(general_config['start_time'])
        self.end_time = str(general_config['end_time'])
        self.max_cache_size = general_config['max_cache_size']
        # 
        self.temp_files = deque(maxlen=self.max_cache_size)
        #
        if WaterSip_method=='WaterSip-HAMSTER':
            self.q_diff_e = config.get('WaterSip-HAMSTER')['q_diff_e']
            self.blh_factor = config.get('WaterSip-HAMSTER')['blh_factor']
            self.P_E_simulation_output_path = config.get('WaterSip-HAMSTER')['P_E_simulation_output_path']
        elif WaterSip_method=='WaterSip-DF-HAMSTER':
            self.q_diff_e = config.get('WaterSip-DF-HAMSTER')['q_diff_e_df']
            self.P_E_simulation_output_path = config.get('WaterSip-DF-HAMSTER')['P_E_simulation_output_path']
            self.DF_file_path = config.get('WaterSip-DF-HAMSTER')['DF_file_path']           


    # %% * * * * * * * * * * MAIN * * * * * * * * * *
    def e_simulation(self, df_method):
        try:
            begin_time = time.time()
            logging.info(f'Step 6 ({self.WaterSip_method}): E simulation...')
            files0 = back_tracking_files(self.partposit_path, self.start_time, self.tracking_days, self.time_span)
            files = get_files(self.partposit_path, self.start_time, self.end_time, self.time_span)
            files = files0[::-1] + files[1:]
            latitude, longitude, gridcell_area = global_gridcell_info(self.output_spatial_resolution, lat_nor=90,
                                                                      lat_sou=-90,
                                                                      lon_lef=-179, lon_rig=180)
            tmp_time = pd.date_range(start=pd.to_datetime(files[0][-14:-4], format='%Y%m%d%H'),
                                     end=pd.to_datetime(files[-1][-14:-4], format='%Y%m%d%H'),
                                     freq='{}h'.format(self.time_span))[:-1]
            E_simulation_mm = np.zeros([np.shape(gridcell_area)[0], np.shape(gridcell_area)[1]])

            for i in range(1, len(files)):
                try:
                    # 
                    df_e = self._read_and_cache_file(files[i], variables=['lon', 'lat', 'q', 'z', 'blh', 'mass'])
                    # 
                    df0 = self._read_and_cache_file(files[i - 1], variables=['lon', 'lat', 'q', 'z', 'blh'])
                    df0 = df0.loc[:, ['lon', 'lat', 'q', 'z', 'blh']]
                    df0 = df0.rename(columns={'lat': 'lat0', 'lon': 'lon0', 'q': 'q0', 'z': 'z0', 'blh': 'blh0'})
                    df_e = df_e.merge(df0, left_index=True, right_index=True)
                    # q_diff threshold filter
                    df_e['q_diff'] = df_e['q'] - df_e['q0']
                    df_e = df_e.drop(columns=['q', 'q0'])
                    df_e = df_e[df_e['q_diff'] > 0]  
                    # 
                    df_e['lat'], df_e['lon'] = midpoint(df_e['lat'].values, df_e['lon'].values, df_e['lat0'].values,
                                                        df_e['lon0'].values)
                    df_e = df_e.drop(columns=['lat0', 'lon0'])
                    # z and abl middle
                    df_e['z'] = (df_e['z'] + df_e['z0']) / 2
                    df_e['blh'] = (df_e['blh'] + df_e['blh0']) / 2
                    df_e = df_e.drop(columns=['z0', 'blh0'])
                    #
                    df_e['lat_round'] = calculate_coordinate_round(df_e['lat'], self.output_spatial_resolution)
                    df_e['lon_round'] = calculate_coordinate_round(df_e['lon'], self.output_spatial_resolution)
                    # df_e = df_e[(df_e['lon_round'] <= 180) & (df_e['lon_round'] >= -179)]     
                    df_e = df_e.drop(columns=['lon', 'lat'])
                    # 
                    if df_method == 'on':
                        BLH_factor = xr.open_dataset(
                            '{}\DF_BLH_factors_{}.nc'.format(self.DF_file_path, tmp_time[i - 1].strftime('%Y%m%d%H')))[
                            'DF_BLH_factors']
                        df_BLH_factor = BLH_factor.to_dataframe(name='BLH_factor').reset_index()
                        df_BLH_factor = df_BLH_factor.query("BLH_factor != 0 and BLH_factor.notna()", engine='numexpr')
                        df_e = pd.merge(df_e, df_BLH_factor, how='left', left_on=['lat_round', 'lon_round'],
                                        right_on=['latitude', 'longitude'])
                        df_e = df_e.drop(columns=['latitude', 'longitude'])
                        df_e.dropna(subset=['BLH_factor'], inplace=True)
                        # df_p = df_p[df_p['RH_threshold'] != 100]
                        df_e = df_e[df_e['z'] <= df_e['blh'] * df_e['BLH_factor']]
                    else:
                        df_e = df_e[df_e['q_diff'] > self.q_diff_e]
                        df_e = df_e[df_e['z'] <= df_e['blh'] * self.blh_factor]
                    # 
                    df_e['e_mass'] = df_e['mass'] * df_e['q_diff']  # kg
                    # 
                    df_grouped_e = df_e.groupby(['lat_round', 'lon_round'])['e_mass'].sum().reset_index()
                    # 
                    lat_idx_e = ((latitude[-1] - df_grouped_e['lat_round']) / self.output_spatial_resolution).astype(int)
                    lon_idx_e = ((df_grouped_e['lon_round'] - longitude[0]) / self.output_spatial_resolution).astype(int)
                    E_simulation_mm[lat_idx_e, lon_idx_e] = df_grouped_e['e_mass']
                    E_simulation_mm = E_simulation_mm / gridcell_area
                    write_to_nc_2d(E_simulation_mm, 'E_simulation_mm',
                                   'E_simulation_mm_{}'.format(tmp_time[i - 1].strftime('%Y%m%d%H')), latitude, longitude,
                                   self.P_E_simulation_output_path)
                    logging.info(f'{files[i - 1].split("partposit_")[-1]} E simulation done!')
                    # del df_e, df0, E_simulation_mm, df_grouped_e, df_BLH_factor
                    # del df_e, df0, df_BLH_factor
                except Exception as e:
                    logging.error(f"Error processing file {files[i]}: {e}")
                    continue

            execution_time = time.time() - begin_time
            logging.info(f"Total duration：{execution_time} seconds")
            logging.info('Step 6 done!')
        except Exception as e:
            logging.error(f"Error in e_simulation: {e}")
            raise

    def _read_and_cache_file(self, file_path, variables):
        for cached_file, cached_data in self.temp_files:
            if cached_file == file_path:
                return cached_data
        # 
        df = readpartposit_to_df(file_path, variables=variables)
        self.temp_files.append((file_path, df))
        return df


if __name__ == "__main__":
    ft = ESimulation()
    ft.e_simulation('on')
