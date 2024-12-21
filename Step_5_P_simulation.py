# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import xarray as xr
from main_functions import get_files, readpartposit_to_df, global_gridcell_info, calculate_RH, midpoint, write_to_nc_2d
from YAMLConfig import YAMLConfig


#%% * * * * * * * * * * MAIN * * * * * * * * * *
def p_simulation(df_method):
    config = YAMLConfig('config.yaml')
    general_config = config.get('general')
    time_span = general_config['time_span']
    # hours, time-step of partposit files
    q_diff_p_simulation = general_config['q_diff_p_simulation']  # kg/kg
    output_spatial_resolution = general_config['output_spatial_resolution']
    start_time = str(general_config['start_time'])
    end_time = str(general_config['end_time'])

    warerSip_HAMSTER_config = config.get('warerSip-HAMSTER')
    q_diff_e = warerSip_HAMSTER_config['q_diff_e']

    watersip_config = config.get('watersip')
    default_q_diff_p = watersip_config['default_q_diff_p']
    default_q_diff_e = watersip_config['default_q_diff_e']
    default_rh_threshold = watersip_config['default_rh_threshold']
    default_blh_factor = watersip_config['default_blh_factor']

    partposit_path = general_config['partposit_path']
    DF_file_path = config.get('warerSip-DF-HAMSTER')['DF_file_path']
    P_E_simulation_output_path = config.get('warerSip-DF-HAMSTER')['P_E_simulation_output_path']

    files = get_files(partposit_path, start_time, end_time, time_span)
    latitude, longitude, gridcell_area = global_gridcell_info(output_spatial_resolution, lat_nor=90, lat_sou=-90, lon_lef=-179, lon_rig=180)
    time = pd.date_range(start=pd.to_datetime(start_time, format='%Y%m%d%H'), end=pd.to_datetime(end_time, format='%Y%m%d%H'), freq='{}H'.format(time_span))[:-1]
    P_simulation_mm = np.zeros([np.shape(gridcell_area)[0], np.shape(gridcell_area)[1]])

    for i in range(1,len(files)): # from 1 start, the files[0] used to calculate diff
        #读取每个时刻数据
        df_p = readpartposit_to_df(files[i], variables=['lon', 'lat', 'q', 't', 'dens', 'mass'])
        calculate_RH(df_p)
        df_p = df_p.drop(columns=['dens','t'])
        # 向前一个时刻
        df0 = readpartposit_to_df(files[i-1], variables=['lon', 'lat', 'q', 't', 'dens'])
        df0 = df0.rename(columns={'lat': 'lat0', 'lon': 'lon0', 'q': 'q0', 't': 't0', 'dens': 'dens0'})
        calculate_RH(df0, RH_name='RH0', dens='dens0', q='q0', t='t0')
        df0 = df0.drop(columns=['dens0','t0'])
        #
        df_p = df_p.merge(df0, left_index=True, right_index=True)
        # 先提取q_diff<0的粒子
        df_p['q_diff'] =  df_p['q'] - df_p['q0']
        df_p = df_p.drop(columns=['q','q0'])
        df_p = df_p[df_p['q_diff'] < q_diff_p_simulation]
        # mid_RH
        df_p['RH'] =  (df_p['RH']+df_p['RH0']) / 2
        df_p = df_p.drop(columns=['RH0'])
        # 计算经纬度中间值
        df_p['lat'], df_p['lon'] = midpoint(df_p['lat'].values, df_p['lon'].values, df_p['lat0'].values, df_p['lon0'].values)
        df_p = df_p.drop(columns=['lat0','lon0'])
        ##
        df_p['lat_round'] = (df_p['lat'] / output_spatial_resolution).round() * output_spatial_resolution
        df_p['lon_round'] = (df_p['lon'] / output_spatial_resolution).round() * output_spatial_resolution
        df_p = df_p.drop(columns=['lat','lon'])
        # df_p = df_p[(df_p['lon_round'] <= 180) & (df_p['lon_round'] >= -179)]
        # 获取相应格点的RH_threshold
        if df_method == 'on':
            RH_threshold = xr.open_dataset('{}\DF_RH_thresholds_{}.nc'.format(DF_file_path, time[i-1].strftime('%Y%m%d%H')))['DF_RH_thresholds']
            df_RH_threshold = RH_threshold.to_dataframe(name='RH_threshold').reset_index()
            df_RH_threshold = df_RH_threshold.query("RH_threshold != 100 and RH_threshold.notna()", engine='numexpr')
            df_p = pd.merge(df_p, df_RH_threshold, how='left', left_on=['lat_round', 'lon_round'], right_on=['latitude', 'longitude'])
            df_p = df_p.drop(columns=['latitude', 'longitude'])
            df_p.dropna(subset=['RH_threshold'], inplace=True)
            #df_p = df_p[df_p['RH_threshold'] != 100]
            df_p = df_p[ df_p['RH'] >= df_p['RH_threshold'] ]
        else:
            df_p = df_p[ df_p['q_diff'] < default_q_diff_p ]
            df_p = df_p[ df_p['RH'] >= default_rh_threshold ]
        # 计算筛选粒子产生的降水
        df_p['p_mass'] = -df_p['mass']*df_p['q_diff'] #kg, *-1 makes P positive
        # 按格点统计并累加p
        df_grouped_p = df_p.groupby(['lat_round', 'lon_round'])['p_mass'].sum().reset_index()
        # p写入0矩阵
        lat_idx_p = ((latitude[-1] - df_grouped_p['lat_round']) / output_spatial_resolution).astype(int)
        lon_idx_p = ((df_grouped_p['lon_round'] - longitude[0]) / output_spatial_resolution).astype(int)
        P_simulation_mm[lat_idx_p, lon_idx_p] = df_grouped_p['p_mass']
        P_simulation_mm = P_simulation_mm/gridcell_area
        write_to_nc_2d(P_simulation_mm, 'P_simulation_mm', 'P_simulation_mm_{}'.format(time[i-1].strftime('%Y%m%d%H')), latitude, longitude, P_E_simulation_output_path)
        print( files[i-1].split("partposit_")[-1], "P simulation done")



