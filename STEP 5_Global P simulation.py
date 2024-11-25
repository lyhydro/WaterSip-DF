# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import xarray as xr
from main_functions import get_files, readpartposit_to_df, global_gridcell_info, calculate_RH, midpoint, write_to_nc

partposit_path = r'F:\FLEXPART\output_2023_July_EAg'
start = '2023070100' # start time for P/E calculation
end = '2023080100' # end time for P/E calculation
time_span = 6 # hours, for partposit files
q_diff_p = -0.0000 #
output_spatial_resolution = 1 # degree

DF_RH_thresholds_path = r'D:\Articles\WaterSip-DF\code-new\temporary\DF_RH_thresholds.nc'
output_path = r'D:\Articles\WaterSip-DF\code-new\temporary'

#%% ********** MAIN **********
files = get_files(partposit_path, start, end, time_span)
latitude, longitude, gridcell_area = global_gridcell_info(output_spatial_resolution, lat_nor=90, lat_sou=-90, lon_lef=-179, lon_rig=180)
time = pd.date_range(start=pd.to_datetime(start, format='%Y%m%d%H'), end=pd.to_datetime(end, format='%Y%m%d%H'), freq='{}H'.format(time_span))[:-1]
DF_RH_thresholds = xr.open_dataset( DF_RH_thresholds_path )[ 'DF_RH_thresholds' ]
P_simulation_mm = np.zeros([len(time), np.shape(gridcell_area)[0], np.shape(gridcell_area)[1]])

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
    df_p = df_p[df_p['q_diff'] < q_diff_p]
    # mid_RH
    df_p['mid_RH'] =  (df_p['RH']+df_p['RH0']) / 2
    df_p = df_p.drop(columns=['RH','RH0'])
    # 计算经纬度中间值
    df_p['lat'], df_p['lon'] = midpoint(df_p['lat'].values, df_p['lon'].values, df_p['lat0'].values, df_p['lon0'].values)
    df_p = df_p.drop(columns=['lat0','lon0'])
    ##
    df_p['lat_round'] = (df_p['lat'] / output_spatial_resolution).round() * output_spatial_resolution
    df_p['lon_round'] = (df_p['lon'] / output_spatial_resolution).round() * output_spatial_resolution  
    df_p = df_p.drop(columns=['lat','lon'])
    # df_p = df_p[(df_p['lon_round'] <= 180) & (df_p['lon_round'] >= -179)]
    # 获取相应格点的RH_threshold
    df_RH_threshold = DF_RH_thresholds[i-1,:,:].to_dataframe(name='RH_threshold').reset_index().drop(columns=['time'])
    df_p = pd.merge(df_p, df_RH_threshold, how='left', left_on=['lat_round', 'lon_round'], right_on=['latitude', 'longitude'])
    df_p = df_p.drop(columns=['latitude', 'longitude'])
    # 删除nan和RH_threshold=100的点
    df_p.dropna(subset=['RH_threshold'], inplace=True)
    df_p = df_p[df_p['RH_threshold'] != 100]
    # 筛选
    df_p = df_p[ df_p['mid_RH'] >= df_p['RH_threshold'] ]   
    # 计算筛选粒子产生的降水
    df_p['p_mass'] = -df_p['mass']*df_p['q_diff'] #kg, *-1 makes P positive    
    # 按格点统计并累加p
    df_grouped_p = df_p.groupby(['lat_round', 'lon_round'])['p_mass'].sum().reset_index()    
    # p写入0矩阵
    lat_idx_p = ((latitude[-1] - df_grouped_p['lat_round']) / output_spatial_resolution).astype(int)
    lon_idx_p = ((df_grouped_p['lon_round'] - longitude[0]) / output_spatial_resolution).astype(int)
    P_simulation_mm[i-1, lat_idx_p, lon_idx_p] = df_grouped_p['p_mass']
    P_simulation_mm[i-1,:,:] = P_simulation_mm[i-1,:,:]/gridcell_area
    print( files[i-1].split("partposit_")[-1], "Global P simulation done")

#%% store output data
write_to_nc(P_simulation_mm, name='P_simulation_mm', time=time, latitude=latitude, longitude=longitude, output_path=output_path)


