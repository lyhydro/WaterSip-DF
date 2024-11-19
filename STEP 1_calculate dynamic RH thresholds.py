# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import xarray as xr
from main_functions import get_files, readpartposit, global_gridcell_info, calculate_RH, midpoint, write_to_nc

partposit_path = r'F:\FLEXPART\output_2023_July_EAg'
observation_file = r'D:\Articles\An optimized bias-correction framework for moisture source-receptor diagnose in Lagrangian models\data\gobal_TP_E_2023_June_July_1h.nc'

start = '2023070100' # start time for P/E calculation
end = '2023080100' # end time for P/E calculation
time_span = 6 # hours, for partposit files
q_diff_p = -0.0000 # 用于提取所有P粒子的阈值，模拟结果对该阈值不敏感；但筛选一下可显著减小计算量，增加运算速度

fast_version = 'on' # 'on' or 'off', if 'off', run slowly

output_spatial_resolution = 1 # degree
output_path = r'D:\Articles\An optimized bias-correction framework for moisture source-receptor diagnose in Lagrangian models\data'

#%% 
time = pd.date_range(start=pd.to_datetime(start, format='%Y%m%d%H'), end=pd.to_datetime(end, format='%Y%m%d%H'), freq='{}H'.format(time_span))
time = time[:-1] # 特定时刻的模拟降水代表的是其前一段时间的降水，这与数据资料中不同，所以最后一时刻删除
P_era5_6h = xr.open_dataset( observation_file )['tp'].resample(time='6H').sum(dim='time')*1000 #将小时数居转换为6小时数居

files = get_files(partposit_path, start, end, time_span)
latitude, longitude, gridcell_area = global_gridcell_info(output_spatial_resolution, lat_nor=90, lat_sou=-90, lon_lef=-179, lon_rig=180)
P_era5_6h = P_era5_6h.where(P_era5_6h > 0, 0) # 将小于等于0的值置0
P_era5_6h = P_era5_6h.sel(time=time)*gridcell_area
RH_threshold_opt = np.zeros([len(time), np.shape(gridcell_area)[0], np.shape(gridcell_area)[1]])

for i in range(1,len(files)): # from 1 start, the files[0] used to calculate diff
    #读取每个时刻数据
    raw = readpartposit(files[i])
    df_p = pd.DataFrame( {'lon':raw['xlon'],'lat':raw['ylat'],'q':raw['qvi'],'t':raw['tti'],'dens':raw['rhoi'],'mass':raw['xmass']}, index=raw['npoint'])    
    calculate_RH(df_p)
    df_p = df_p.drop(columns=['t','dens'])
    # 向前一个时刻
    raw0 = readpartposit(files[i-1]) 
    df0 = pd.DataFrame({'lon0':raw0['xlon'],'lat0':raw0['ylat'],'q0':raw0['qvi'],'t0':raw0['tti'],'dens0':raw0['rhoi']}, index=raw0['npoint'])
    calculate_RH(df0, RH_name='RH0', dens='dens0', q='q0', t='t0')
    df0 = df0.drop(columns=['t0','dens0'])
    #
    df_p = df_p.merge(df0, left_index=True, right_index=True) 
    # q_diff 筛选
    df_p['q_diff'] =  df_p['q'] - df_p['q0']
    df_p = df_p.drop(columns=['q','q0'])
    df_p = df_p[df_p['q_diff'] < q_diff_p]  #筛选   
    # 计算经纬度中间值
    df_p['lat'], df_p['lon'] = midpoint(df_p['lat'].values, df_p['lon'].values, df_p['lat0'].values, df_p['lon0'].values)
    df_p = df_p.drop(columns=['lat0','lon0'])
    # RHm
    df_p['RHm'] =  (df_p['RH']+df_p['RH0']) / 2
    df_p = df_p.drop(columns=['RH','RH0'])
    #
    df_p['lat_round'] = (df_p['lat'] / output_spatial_resolution).round() * output_spatial_resolution
    df_p['lon_round'] = (df_p['lon'] / output_spatial_resolution).round() * output_spatial_resolution
    df_p = df_p.drop(columns=['lon','lat'])
    # 删除经度大于180和小于-179的行，保持于ERA5一致
    df_p = df_p[(df_p['lon_round'] <= 180) & (df_p['lon_round'] >= -179)]
    # 
    if fast_version == 'on':
        # p_simulation = np.zeros([101,len(latitude),len(longitude)]) # 从0%-100%的rh阈值所模拟的101层的全球p_simulation数据
        # for j in range(0,101):
        #     df_p0 = df_p.copy()
        #     df_p0 = df_p0[ df_p0['RHm'] > j ]
        #     # 计算筛选粒子产生的降水
        #     df_p0['p_mass'] = -df_p0['mass']*df_p0['q_diff'] #kg, *-1 makes P positive    
        #     # 按格点统计并累加p
        #     df_grouped_p = df_p0.groupby(['lat_round', 'lon_round'])['p_mass'].sum().reset_index()    
        #     # p写入0矩阵
        #     lat_idx_p = ((latitude[-1] - df_grouped_p['lat_round']) / output_spatial_resolution).astype(int)
        #     lon_idx_p = ((df_grouped_p['lon_round'] - longitude[0]) / output_spatial_resolution).astype(int)       
        #     p_simulation[j, lat_idx_p, lon_idx_p] = df_grouped_p['p_mass']
        #     print(j,'-', end='')
        # p_simulation[p_simulation==0] = np.nan # nan代表模拟降水为0的点
        # # 在每个格点与实测对比，找到最优的RH阈值
        # for h in range(0,len(latitude)):
        #     for l in range(0,len(longitude)):
        #         obs0 = P_era5_6h[i-1,h,l].values
        #         p_simulation0 = p_simulation[:,h,l]
        #         if np.all(np.isnan( p_simulation0 )):
        #             RH_threshold_opt[i-1,h,l] = np.nan
        #         elif obs0==0:
        #             RH_threshold_opt[i-1,h,l] = 100 # 无观测降水时，阈值100%
        #         else:
        #             array_test = p_simulation0 - obs0
        #             RH_threshold_opt[i-1,h,l] = np.nanargmin(np.abs(array_test))
        #     print('.', end='')
        # print( files[i-1].split("partposit_")[-1], "done ！！！")
        
        rh_span = np.arange(0,100.1,0.2)
        p_simulation = np.zeros([len(rh_span),len(latitude),len(longitude)]) # 从0%-100%的rh阈值所模拟的1001层(跨度0.1%)的全球p_simulation数据
        for j in range(0,len(rh_span)):
            df_p0 = df_p.copy()
            df_p0 = df_p0[ df_p0['RHm'] > rh_span[j] ]
            # 计算筛选粒子产生的降水
            df_p0['p_mass'] = -df_p0['mass']*df_p0['q_diff'] #kg, *-1 makes P positive    
            # 按格点统计并累加p
            df_grouped_p = df_p0.groupby(['lat_round', 'lon_round'])['p_mass'].sum().reset_index()    
            # p写入0矩阵
            lat_idx_p = ((latitude[-1] - df_grouped_p['lat_round']) / output_spatial_resolution).astype(int)
            lon_idx_p = ((df_grouped_p['lon_round'] - longitude[0]) / output_spatial_resolution).astype(int)       
            p_simulation[j, lat_idx_p, lon_idx_p] = df_grouped_p['p_mass']
            print(rh_span[j],'-', end='')
        p_simulation[p_simulation==0] = np.nan # nan代表模拟降水为0的点
        # 在每个格点与实测对比，找到最优的RH阈值
        for h in range(0,len(latitude)):
            for l in range(0,len(longitude)):
                obs0 = P_era5_6h[i-1,h,l].values
                p_simulation0 = p_simulation[:,h,l]
                if np.all(np.isnan( p_simulation0 )):
                    RH_threshold_opt[i-1,h,l] = np.nan
                elif obs0==0:
                    RH_threshold_opt[i-1,h,l] = 100 # 无观测降水时，阈值100%
                else:
                    array_test = p_simulation0 - obs0
                    RH_threshold_opt[i-1,h,l] = rh_span[ np.nanargmin(np.abs(array_test)) ]
            print('.', end='')
        print( files[i-1].split("partposit_")[-1], "done ！！！")
        
    if fast_version == 'off':
        for lat in latitude:
            for lon in longitude:
                obs = P_era5_6h[i-1,:,:].sel(latitude=lat,longitude=lon).values
                selected_data = df_p.query("lat_round == @lat and lon_round == @lon").sort_values(by='RHm', ascending=False)
                if len(selected_data) == 0:
                    RH_threshold_opt[i-1,np.where(latitude==lat),np.where(longitude==lon)] = np.nan # nan代表模拟降水为0的点
                elif obs==0:
                    RH_threshold_opt[i-1,np.where(latitude==lat),np.where(longitude==lon)] = 100
                else:
                    row_opt = np.argmin( np.abs( (-selected_data['mass']*selected_data['q_diff']).cumsum() - obs) )
                    RH_opt = selected_data['RHm'].iloc[row_opt]
                    RH_threshold_opt[i-1,np.where(latitude==lat),np.where(longitude==lon)] = RH_opt
            print('.', end='')
        print( files[i-1].split("partposit_")[-1], "done ！！！")
    
#%% store output data
write_to_nc(RH_threshold_opt, name='RH_threshold_opt', time=time, latitude=latitude, longitude=longitude, output_path=output_path)

