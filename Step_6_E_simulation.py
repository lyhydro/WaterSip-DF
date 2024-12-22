# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import xarray as xr
from main_functions import get_files, back_tracking_files, readpartposit_to_df, global_gridcell_info, midpoint, write_to_nc_2d
from YAMLConfig import YAMLConfig


# %% * * * * * * * * * * MAIN * * * * * * * * * *
def e_simulation(df_method):
    config = YAMLConfig('config.yaml')
    general_config = config.get('General')
    partposit_path = general_config['partposit_path']
    tracking_days = general_config['tracking_days']
    time_span = general_config['time_span']
    output_spatial_resolution = general_config['output_spatial_resolution']
    start_time = str(general_config['start_time'])
    end_time = str(general_config['end_time'])

    warerSip_HAMSTER_config = config.get('WaterSip-HAMSTER')
    q_diff_p = warerSip_HAMSTER_config['q_diff_p']
    q_diff_e = warerSip_HAMSTER_config['q_diff_e']
    rh_threshold = warerSip_HAMSTER_config['rh_threshold']
    blh_factor = warerSip_HAMSTER_config['blh_factor']
    P_E_simulation_output_path = warerSip_HAMSTER_config['P_E_simulation_output_path']
    
    DF_file_path = config.get('WaterSip-DF-HAMSTER')['DF_file_path']

    files0 = back_tracking_files(partposit_path, start_time, tracking_days, time_span)
    files = get_files(partposit_path, start_time, end_time, time_span)
    files = files0[:-1] + files
    latitude, longitude, gridcell_area = global_gridcell_info(output_spatial_resolution, lat_nor=90, lat_sou=-90,
                                                              lon_lef=-179, lon_rig=180)
    time = pd.date_range(start=pd.to_datetime(files[0][-14:-4], format='%Y%m%d%H'),
                         end=pd.to_datetime(files[-1][-14:-4], format='%Y%m%d%H'), freq='{}H'.format(time_span))[:-1]
    E_simulation_mm = np.zeros([np.shape(gridcell_area)[0], np.shape(gridcell_area)[1]])

    for i in range(1, len(files)):  # from 1 start, the files[0] used to calculate diff
        # 读取每个时刻数据
        df_e = readpartposit_to_df(files[i], variables=['lon', 'lat', 'q', 'z', 'blh', 'mass'])
        # 向前一个时刻
        df0 = readpartposit_to_df(files[i - 1], variables=['lon', 'lat', 'q', 'z', 'blh'])
        df0 = df0.rename(columns={'lat': 'lat0', 'lon': 'lon0', 'q': 'q0', 'z': 'z0', 'blh': 'blh0'})
        df_e = df_e.merge(df0, left_index=True, right_index=True)
        # q_diff threshold filter
        df_e['q_diff'] = df_e['q'] - df_e['q0']
        df_e = df_e.drop(columns=['q', 'q0'])
        df_e = df_e[df_e['q_diff'] > 0]  # 比湿变化阈值筛选
        # 计算经纬度中间值
        df_e['lat'], df_e['lon'] = midpoint(df_e['lat'].values, df_e['lon'].values, df_e['lat0'].values,
                                            df_e['lon0'].values)
        df_e = df_e.drop(columns=['lat0', 'lon0'])
        # z and abl middle
        df_e['z'] = (df_e['z'] + df_e['z0']) / 2
        df_e['blh'] = (df_e['blh'] + df_e['blh0']) / 2
        df_e = df_e.drop(columns=['z0', 'blh0'])
        #
        df_e['lat_round'] = (df_e['lat'] / output_spatial_resolution).round() * output_spatial_resolution
        df_e['lon_round'] = (df_e['lon'] / output_spatial_resolution).round() * output_spatial_resolution
        # df_e = df_e[(df_e['lon_round'] <= 180) & (df_e['lon_round'] >= -179)]     # 删除经度大于180和小于-179的行，保持于ERA5一致
        df_e = df_e.drop(columns=['lon', 'lat'])
        # 获取相应格点的abl_factor
        if df_method == 'on':
            BLH_factor = \
            xr.open_dataset('{}\DF_BLH_factors_{}.nc'.format(DF_file_path, time[i - 1].strftime('%Y%m%d%H')))[
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
            df_e = df_e[df_e['q_diff'] > q_diff_e]
            df_e = df_e[df_e['z'] <= df_e['blh'] * blh_factor]
        # 计算筛选粒子产生的蒸发
        df_e['e_mass'] = df_e['mass'] * df_e['q_diff']  # kg
        # 按格点统计并累加e
        df_grouped_e = df_e.groupby(['lat_round', 'lon_round'])['e_mass'].sum().reset_index()
        # e写入0矩阵
        lat_idx_e = ((latitude[-1] - df_grouped_e['lat_round']) / output_spatial_resolution).astype(int)
        lon_idx_e = ((df_grouped_e['lon_round'] - longitude[0]) / output_spatial_resolution).astype(int)
        E_simulation_mm[lat_idx_e, lon_idx_e] = df_grouped_e['e_mass']
        E_simulation_mm = E_simulation_mm / gridcell_area
        write_to_nc_2d(E_simulation_mm, 'E_simulation_mm',
                       'E_simulation_mm_{}'.format(time[i - 1].strftime('%Y%m%d%H')), latitude, longitude,
                       P_E_simulation_output_path)
        print(files[i - 1].split("partposit_")[-1], "E simulation done")
