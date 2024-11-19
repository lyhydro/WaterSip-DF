# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import xarray as xr
from main_functions import get_files, readpartposit, global_gridcell_info, calculate_RH, midpoint, write_to_nc

partposit_path = r'F:\FLEXPART\output_2023_July_EAg'
start = '2023070100' # start time for P/E calculation
end = '2023080100' # end time for P/E calculation
time_span = 6 # hours, for partposit files
q_diff_p = -0.0000 #
output_spatial_resolution = 1 # degree

RH_threshold_opt_path = r'D:\Articles\An optimized bias-correction framework for moisture source-receptor diagnose in Lagrangian models\data\RH_threshold_opt.nc'
RH_threshold_opt = xr.open_dataset( RH_threshold_opt_path )[ 'RH_threshold_opt' ]

output_path = r'D:\Articles\An optimized bias-correction framework for moisture source-receptor diagnose in Lagrangian models\data'

#%% ********** MAIN **********
files = get_files(partposit_path, start, end, time_span)
latitude, longitude, gridcell_area = global_gridcell_info(output_spatial_resolution, lat_nor=90, lat_sou=-90, lon_lef=-179, lon_rig=180)
time = pd.date_range(start=pd.to_datetime(start, format='%Y%m%d%H'), end=pd.to_datetime(end, format='%Y%m%d%H'), freq='{}H'.format(time_span))
time = time[:-1] # 特定时刻的模拟降水代表的是其前一段时间的降水，这与数据资料中不同，所以最后一时刻删除
P_simulation_mm = np.zeros([len(time), np.shape(gridcell_area)[0], np.shape(gridcell_area)[1]])

for i in range(1,len(files)): # from 1 start, the files[0] used to calculate diff
    #读取每个时刻数据
    raw = readpartposit(files[i])
    df_p = pd.DataFrame( {'lon':raw['xlon'],'lat':raw['ylat'],'q':raw['qvi'],'t':raw['tti'],'dens':raw['rhoi'],'mass':raw['xmass']}, index=raw['npoint'])    
    calculate_RH(df_p)['RH']
    df_p = df_p.drop(columns=['dens','t'])
    # 向前一个时刻
    raw0 = readpartposit(files[i-1]) 
    df0 = pd.DataFrame({'lon0':raw0['xlon'],'lat0':raw0['ylat'],'q0':raw0['qvi'],'t0':raw0['tti'],'dens0':raw0['rhoi']}, index=raw0['npoint'])      
    calculate_RH(df0, RH_name='RH0', dens='dens0', q='q0', t='t0')
    df0 = df0.drop(columns=['dens0','t0'])
    #
    df_p = df_p.merge(df0, left_index=True, right_index=True) 
    # 先提取q_diff<0的粒子
    df_p['q_diff'] =  df_p['q'] - df_p['q0']
    df_p = df_p.drop(columns=['q','q0'])
    df_p = df_p[df_p['q_diff'] < q_diff_p]
    # RHm
    df_p['RHm'] =  (df_p['RH']+df_p['RH0']) / 2
    df_p = df_p.drop(columns=['RH','RH0'])
    # 计算经纬度中间值
    df_p['lat'], df_p['lon'] = midpoint(df_p['lat'].values, df_p['lon'].values, df_p['lat0'].values, df_p['lon0'].values)
    df_p = df_p.drop(columns=['lat0','lon0'])
    ##
    df_p['lat_round'] = (df_p['lat'] / output_spatial_resolution).round() * output_spatial_resolution
    df_p['lon_round'] = (df_p['lon'] / output_spatial_resolution).round() * output_spatial_resolution  
    df_p = df_p.drop(columns=['lat','lon'])
    # 删除经度大于180和小于-179的行，保持于ERA5一致
    df_p = df_p[(df_p['lon_round'] <= 180) & (df_p['lon_round'] >= -179)]
    # 获取相应格点的rh_threshold
    df_RH_threshold_opt = RH_threshold_opt[i-1,:,:].to_dataframe(name='rh_threshold').reset_index().drop(columns=['time'])
    df_p = pd.merge(df_p, df_RH_threshold_opt, how='left', left_on=['lat_round', 'lon_round'], right_on=['latitude', 'longitude'])
    df_p = df_p.drop(columns=['latitude', 'longitude'])
    # 删除nan和rh_threshold=100的点
    df_p.dropna(subset=['rh_threshold'], inplace=True)
    df_p = df_p[df_p['rh_threshold'] != 100]
    # 筛选
    df_p = df_p[ df_p['RHm'] >= df_p['rh_threshold'] ]   
    # 计算筛选粒子产生的降水
    df_p['p_mass'] = -df_p['mass']*df_p['q_diff'] #kg, *-1 makes P positive    
    # 按格点统计并累加p
    df_grouped_p = df_p.groupby(['lat_round', 'lon_round'])['p_mass'].sum().reset_index()    
    # p写入0矩阵
    lat_idx_p = ((latitude[-1] - df_grouped_p['lat_round']) / output_spatial_resolution).astype(int)
    lon_idx_p = ((df_grouped_p['lon_round'] - longitude[0]) / output_spatial_resolution).astype(int)
    P_simulation_mm[i-1, lat_idx_p, lon_idx_p] = df_grouped_p['p_mass']
    P_simulation_mm[i-1,:,:] = P_simulation_mm[i-1,:,:]/gridcell_area
    print( files[i-1].split("partposit_")[-1], "P simulation done")

#%% store output data
write_to_nc(P_simulation_mm, name='P_simulation_mm_by_RH_threshold_opt', time=time, latitude=latitude, longitude=longitude, output_path=output_path)


