# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import xarray as xr
from main_functions import get_files, readpartposit, global_gridcell_info, midpoint, write_to_nc

partposit_path = r'F:\FLEXPART\output_2023_July_EAg'
observation_file = r'D:\Articles\An optimized bias-correction framework for moisture source-receptor diagnose in Lagrangian models\data\gobal_TP_E_2023_June_July_1h.nc'

start = '2023070100' # start time for P/E calculation
end = '2023080100' # end time for P/E calculation
time_span = 6 # hours, for partposit files
q_diff_e = 0.0000 # 用于提取所有E粒子；但筛选一下可显著减小计算量

fast_but_rough_version = 'on' # only support 'on'

output_spatial_resolution = 1 # degree
output_path = r'D:\Articles\An optimized bias-correction framework for moisture source-receptor diagnose in Lagrangian models\data'

#%% 
time = pd.date_range(start=pd.to_datetime(start, format='%Y%m%d%H'), end=pd.to_datetime(end, format='%Y%m%d%H'), freq='{}H'.format(time_span))
time = time[:-1] # 特定时刻的模拟降水代表的是其前一段时间的降水，这与数据资料中不同，所以最后一时刻删除
E_era5_6h = -xr.open_dataset( observation_file )['e'].resample(time='6H').sum(dim='time')*1000 #将小时数居转换为6小时数居

files = get_files(partposit_path, start, end, time_span)
latitude, longitude, gridcell_area = global_gridcell_info(output_spatial_resolution, lat_nor=90, lat_sou=-90, lon_lef=-179, lon_rig=180)
E_era5_6h = E_era5_6h.where(E_era5_6h > 0, 0) # 将小于等于0的值置0
E_era5_6h = E_era5_6h.sel(time=time)*gridcell_area

ABL_factor_opt = np.zeros([len(time), np.shape(gridcell_area)[0], np.shape(gridcell_area)[1]])

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
    # 
    
    if fast_but_rough_version == 'on':
        abl_factors = np.arange(0.2,10.0,0.02) # 设定所有用来尝试的abl_factor
        e_simulation = np.zeros([len(abl_factors),len(latitude),len(longitude)]) # 从0%-100%的rh阈值所模拟的101层的全球p_simulation数据
        for j in range(0, len(abl_factors)): 
            df_e0 = df_e.copy()
            df_e0 = df_e0[df_e0['z'] < abl_factors[j]*df_e0['abl']]
            # 计算筛选粒子产生的蒸发            
            df_e0['e_mass'] = df_e0['mass']*df_e0['q_diff'] # kg
            df_grouped_e = df_e0.groupby(['lat_round', 'lon_round'])['e_mass'].sum().reset_index()    
            # e写入0矩阵
            lat_idx_e = ((latitude[-1] - df_grouped_e['lat_round']) / output_spatial_resolution).astype(int)
            lon_idx_e = ((df_grouped_e['lon_round'] - longitude[0]) / output_spatial_resolution).astype(int)       
            e_simulation[j, lat_idx_e, lon_idx_e] = df_grouped_e['e_mass']
            print(abl_factors[j],'-', end='')
        # e_simulation[e_simulation==0] = np.nan # nan代表模拟为0的点
        # 在每个格点与实测对比，找到最优的abl_factor
        for h in range(0,len(latitude)):
            for l in range(0,len(longitude)):
                obs0 = E_era5_6h[i-1,h,l].values
                e_simulation0 = e_simulation[:,h,l]
                if e_simulation0[-1]==0:
                    ABL_factor_opt[i-1,h,l] = np.nan # 模拟蒸发为0，说明没有蒸发粒子，系数选最大abl_factors[-1]还是用nan？
                elif obs0==0:
                    ABL_factor_opt[i-1,h,l] = abl_factors[0] # 测降蒸发为0，系数选最小
                else:
                    array_test = e_simulation0 - obs0
                    row_test = np.nanargmin(np.abs(array_test)) 
                    ABL_factor_opt[i-1,h,l] = abl_factors[row_test]          
            print('.', end='')
        print( files[i-1].split("partposit_")[-1], "done ！！！")
    #
    if fast_but_rough_version == 'off':
        print( 'none' )
        break
    
#%% store output data
write_to_nc(ABL_factor_opt, name='ABL_factor_opt', time=time, latitude=latitude, longitude=longitude, output_path=output_path)
