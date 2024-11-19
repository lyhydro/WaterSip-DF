# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import xarray as xr
from main_functions import get_files, readpartposit, global_gridcell_info, midpoint, write_to_nc

partposit_path = r'F:\FLEXPART\output_2023_July_EAg'
start = '2023070100' # start time for P/E calculation
end = '2023080100' # end time for P/E calculation
time_span = 6 # hours, for partposit files
q_diff_e = 0.0000 # kg/kg
output_spatial_resolution = 1 # degree

ABL_factor_opt_path = r'D:\Articles\An optimized bias-correction framework for moisture source-receptor diagnose in Lagrangian models\data\ABL_factor_opt.nc'
ABL_factor_opt = xr.open_dataset( ABL_factor_opt_path )[ 'ABL_factor_opt' ]

output_path = r'D:\Articles\An optimized bias-correction framework for moisture source-receptor diagnose in Lagrangian models\data'

#%% ********** MAIN **********
files = get_files(partposit_path, start, end, time_span)
latitude, longitude, gridcell_area = global_gridcell_info(output_spatial_resolution, lat_nor=90, lat_sou=-90, lon_lef=-179, lon_rig=180)
time = pd.date_range(start=pd.to_datetime(start, format='%Y%m%d%H'), end=pd.to_datetime(end, format='%Y%m%d%H'), freq='{}H'.format(time_span))
time = time[:-1] # 特定时刻的模拟降水代表的是其前一段时间的降水，这与数据资料中不同，所以最后一时刻删除
E_simulation_mm = np.zeros([len(time), np.shape(gridcell_area)[0], np.shape(gridcell_area)[1]])

for i in range(1,len(files)): # from 1 start, the files[0] used to calculate diff
    #读取每个时刻数据
    raw = readpartposit(files[i])
    df_e = pd.DataFrame( {'lon':raw['xlon'],'lat':raw['ylat'],'q':raw['qvi'],'z':raw['ztra'], 'abl':raw['hmixi'],'mass':raw['xmass']}, index=raw['npoint'])    
    # 向前一个时刻
    raw0 = readpartposit(files[i-1]) 
    df0 = pd.DataFrame({'lon0':raw0['xlon'],'lat0':raw0['ylat'],'q0':raw0['qvi'],'z0':raw0['ztra'], 'abl0':raw0['hmixi']}, index=raw0['npoint'])
    df_e = df_e.merge(df0, left_index=True, right_index=True) 
    # q_diff threshold filter
    df_e['q_diff'] =  df_e['q'] - df_e['q0']
    df_e = df_e.drop(columns=['q','q0'])
    df_e = df_e[df_e['q_diff'] > q_diff_e] #比湿变化阈值筛选   
    # 计算经纬度中间值
    df_e['lat'], df_e['lon'] = midpoint(df_e['lat'].values, df_e['lon'].values, df_e['lat0'].values, df_e['lon0'].values)
    df_e = df_e.drop(columns=['lat0','lon0'])    
    # z and abl middle
    df_e['z'] =  (df_e['z']+df_e['z0']) / 2
    df_e['abl'] =  (df_e['abl']+df_e['abl0']) / 2
    df_e = df_e.drop(columns=['z0','abl0'])    
    #
    df_e['lat_round'] = (df_e['lat'] / output_spatial_resolution).round() * output_spatial_resolution
    df_e['lon_round'] = (df_e['lon'] / output_spatial_resolution).round() * output_spatial_resolution
    df_e = df_e[(df_e['lon_round'] <= 180) & (df_e['lon_round'] >= -179)]     # 删除经度大于180和小于-179的行，保持于ERA5一致
    df_e = df_e.drop(columns=['lon','lat'])
    # 获取相应格点的abl_factor
    df_ABL_factor_opt = ABL_factor_opt[i-1,:,:].to_dataframe(name='abl_factor').reset_index().drop(columns=['time'])
    df_e = pd.merge(df_e, df_ABL_factor_opt, how='left', left_on=['lat_round', 'lon_round'], right_on=['latitude', 'longitude'])    
    # 删除nan的点
    df_e.dropna(subset=['abl_factor'], inplace=True)
    # 筛选
    df_e = df_e[ df_e['z'] < df_e['abl_factor']*df_e['abl'] ]      
    # 计算筛选粒子产生的蒸发
    df_e['e_mass'] = df_e['mass']*df_e['q_diff'] # kg  
    # 按格点统计并累加e
    df_grouped_e = df_e.groupby(['lat_round', 'lon_round'])['e_mass'].sum().reset_index()    
    # e写入0矩阵
    lat_idx_e = ((latitude[-1] - df_grouped_e['lat_round']) / output_spatial_resolution).astype(int)
    lon_idx_e = ((df_grouped_e['lon_round'] - longitude[0]) / output_spatial_resolution).astype(int)
    E_simulation_mm[i-1, lat_idx_e, lon_idx_e] = df_grouped_e['e_mass']
    E_simulation_mm[i-1,:,:] = E_simulation_mm[i-1,:,:]/gridcell_area
    print( files[i-1].split("partposit_")[-1], "E simulation done")

#%% store output data
write_to_nc(E_simulation_mm, name='E_simulation_mm_by_ABL_factor_opt', time=time, latitude=latitude, longitude=longitude, output_path=output_path)


