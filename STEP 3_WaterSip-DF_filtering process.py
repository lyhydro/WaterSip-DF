# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
import xarray as xr
from main_functions import readpartposit, get_files, from_shp_get_parts, calculate_RH, midpoint, back_rtacking_files, writepkl

#%% * * * * * * * * * * * * * * * * * * * * * setting * * * * * * * * * * * * * * * * * * * * * 
target_shp = r'D:\Articles\An optimized bias-correction framework for moisture source-receptor diagnose in Lagrangian models\YRB\shp\boundary.shp'
start = '2023072600' # the diagnose start time (partposit file start time - time span)
end = '2023080100' 
time_span = 6
tracking_days = 15
flexpart_output_file = r'F:\FLEXPART\output_2023_July_EAg'
temporary_file_path = r'D:\Articles\An optimized bias-correction framework for moisture source-receptor diagnose in Lagrangian models\YRB\temporary_output'
output_spatial_resolution = 1

dynamic_rh_threshold = 'on'  # 'on' or 'off'
dynamic_abl_factor = 'on'  # 'on' or 'off'

q_diff_p = -0.0000 # kg/kg
q_diff_e = 0.0000 # kg/kg
dynamic_rh_threshold_file = r'D:\Articles\An optimized bias-correction framework for moisture source-receptor diagnose in Lagrangian models\data\RH_threshold_opt.nc'
dynamic_abl_factor_file = r'D:\Articles\An optimized bias-correction framework for moisture source-receptor diagnose in Lagrangian models\data\ABL_factor_opt.nc'

default_q_diff_p = -0.0002 # kg/kg
default_q_diff_e = 0.0002 # kg/kg
default_rh_threshold = 80
default_abl_factor = 1.5

#%% * * * * * * * * * * * * * * * * * * * * * MAIN - found all P particles * * * * * * * * * * * * * * * * * * * * * 
files = get_files(flexpart_output_file, start, end, time_span)
RH_threshold_opt = xr.open_dataset( dynamic_rh_threshold_file )[ 'RH_threshold_opt' ]
#筛选在目标区产生降水的例子，包括部分轨迹重复的不同时刻的粒子
all_parts = []; start_times = []
for i in range(1,len(files)): #这里从1开始，是因为get_release_files多取了向前一步，用于计算q_diff   
    raw = readpartposit(files[i])
    #
    df_p = pd.DataFrame( {'lon':raw['xlon'],'lat':raw['ylat'],'z':raw['ztra'],'q':raw['qvi'], 'abl':raw['hmixi'],'t':raw['tti'],'dens':raw['rhoi'],'mass':raw['xmass']}, index=raw['npoint'])    
    calculate_RH(df_p)
    # 向前一个时刻
    raw0 = readpartposit(files[i-1]) 
    df0 = pd.DataFrame({'lon0':raw0['xlon'],'lat0':raw0['ylat'],'q0':raw0['qvi'],'t0':raw0['tti'],'dens0':raw0['rhoi']}, index=raw0['npoint'])
    calculate_RH(df0, RH_name='RH0', dens='dens0', q='q0', t='t0')
    df_p = df_p.merge(df0, left_index=True, right_index=True) 
    df_p = df_p.drop(columns=['dens','dens0', 't', 't0'])
    # q_diff threshold filter
    df_p['q_diff'] =  df_p['q'] - df_p['q0']
    df_p = df_p.drop(columns=['q0'])
    df_p = df_p[df_p['q_diff'] < q_diff_p] #比湿变化阈值筛选   
    # 计算经纬度中间值
    df_p['lat'], df_p['lon'] = midpoint(df_p['lat'].values, df_p['lon'].values, df_p['lat0'].values, df_p['lon0'].values)
    df_p = df_p.drop(columns=['lat0','lon0'])
    # 筛选研究区粒子
    df_p = from_shp_get_parts(df_p,target_shp)
    # RH_mid
    df_p['RH'] =  (df_p['RH']+df_p['RH0']) / 2
    df_p = df_p.drop(columns=['RH0'])    
    ##
    df_p['lat_round'] = (df_p['lat'] / output_spatial_resolution).round() * output_spatial_resolution
    df_p['lon_round'] = (df_p['lon'] / output_spatial_resolution).round() * output_spatial_resolution  
    df_p = df_p[(df_p['lon_round'] <= 180) & (df_p['lon_round'] >= -179)] # 删除经度大于180和小于-179的行，于rh_threshold_opt文件保持一致
    ## first correct by rh_threshold
    if dynamic_rh_threshold == 'on':
        time_dt = pd.to_datetime(files[i-1][-14:-4], format='%Y%m%d%H') # 找到相同的时间
        RH_threshold = RH_threshold_opt.sel(time=time_dt)
        df_RH_threshold = RH_threshold.to_dataframe(name='RH_threshold').reset_index().drop(columns=['time'])
        original_index = df_p.index # 保存原索引，因为下一个合并不是基于index的，所以会重置索引
        df_p = pd.merge(df_p, df_RH_threshold, how='left', left_on=['lat_round', 'lon_round'], right_on=['latitude', 'longitude'])  
        df_p.index = original_index # 恢复索引
        df_p.dropna(subset=['RH_threshold'], inplace=True) # 删除rh_threshold=nan的点
        df_p = df_p[df_p['RH_threshold'] != 100] # 删除rh_threshold=100的点
        df_p = df_p[ df_p['RH'] >= df_p['RH_threshold'] ]   
        df_p = df_p.drop(columns=['latitude', 'longitude', 'RH', 'RH_threshold'])             
    else: # use default rh_threshold
        df_p = df_p[df_p['q_diff'] < default_q_diff_p]
        df_p = df_p[df_p['RH'] > default_rh_threshold]
    # calculate p simulation
    df_p['p_mass'] =  -df_p['mass']*df_p['q_diff']      
    all_parts.append(df_p)
    start_times.append(files[i-1][-14:-4])
    if len(df_p)==0:
        print("!!!!! no parts found at this time-step !!!!!")
    print('selecting P parts in ',files[i-1][-14:-1])

#%% * * * * * * * * * * * * * * * * * * * * * MAIN - tracking processes * * * * * * * * * * * * * * * * * * * * * 
ABL_factor_opt = xr.open_dataset( dynamic_abl_factor_file )[ 'ABL_factor_opt' ]
df_ABL_factor = ABL_factor_opt.to_dataframe(name='ABL_factor').reset_index()
#包含源区蒸发阈值筛选、为每个时刻的追踪粒子生成相应追踪信息表格，并输出为临时文件便于查看
for i in range(0,len(start_times)):
    result = []
    files_track = back_rtacking_files(flexpart_output_file, start_times[i], tracking_days, time_span)
    result.append(all_parts[i]) # result的第一个数据即以上某一时刻的all_parts数据，已经求了middle
    for f in files_track[::-1]:
        raw = readpartposit(f)
        #边界层高度为hmixi，1.5为sodemann推荐的系数。水汽uptake发生在边界层以内才算源区，但降水不一定发生在边界层内！！！
        df_raw = pd.DataFrame( {'lon':raw['xlon'],'lat':raw['ylat'],'z':raw['ztra'],'q':raw['qvi'],'abl':raw['hmixi'],'mass':raw['xmass']}, index=raw['npoint'])
        result.append( df_raw.loc[df_raw.index.intersection(result[-1].index)] )
        print('.',end='')
    print('\n',start_times[i],'P particles get backtrack files done')
    
    ## if need temporary_file, use the following 3 lines
    # with open("{}\{}-{}.pkl".format(temporary_file_path,start_times[i],time_span), 'wb') as file:
    #     pickle.dump(result, file)
    # print('\n',start_times[i],'write to temporary_file')
    
    ## establish particle tracking table
    indexs = result[0].index 
    df = []; n = 1
    track_times = pd.to_datetime(start_times[i], format='%Y%m%d%H') - pd.Timedelta(hours=time_span) * np.arange(len(result)-1)
    for index in indexs:
        #每个粒子（index）生成一个独立的按时间降序排列的dataframe
        df_index = []
        df_index.append( result[0].loc[index][['lon','lat','z','q','abl','q_diff','p_mass']] )
        for j in range(1,len(result)):
            if index in result[j].index:
                df_index.append( result[j].loc[index][['lon','lat','z','q','abl','mass']] ) # j=0的值已经求了middle
            else:
                df_index.append( df_index[-1].apply(lambda x: np.nan) ) #若粒子丢失，用nan代表
        df_index = pd.DataFrame(df_index) 
        # lat_mid, lon_mid, z_mid, q_diff, and abl_mid
        df_index['lat'][1:-1], df_index['lon'][1:-1] = midpoint(df_index['lat'][1:-1].values, df_index['lon'][1:-1].values, df_index['lat'][2:].values, df_index['lon'][2:].values)
        df_index['z'][1:-1] = (df_index['z'][1:-1].values + df_index['z'][2:].values) / 2
        df_index['q_diff'][1:-1] = df_index['q'][1:-1].values - df_index['q'][2:].values
        df_index['abl'][1:-1] = (df_index['abl'][1:-1].values + df_index['abl'][2:].values) / 2
        df_index = df_index.iloc[:-1]
        #        
        df_index['lat_round'] = (df_index['lat'] / output_spatial_resolution).round() * output_spatial_resolution
        df_index['lon_round'] = (df_index['lon'] / output_spatial_resolution).round() * output_spatial_resolution 
        # 
        if dynamic_abl_factor == 'on': # use dynamic ABL_factor
            df_index['start_time'] = track_times
            df_index = pd.merge(df_index, df_ABL_factor, how='left', left_on=['lat_round', 'lon_round', 'start_time'], right_on=['latitude', 'longitude', 'time'])         
            df_index = df_index.drop(columns=['latitude', 'longitude', 'start_time', 'time'])    
            #通过阈值筛选蒸发区域，计算t<0的降水贡献比率f
            df_index['f0'] = np.where( (df_index['q_diff']>q_diff_e) & ( df_index['z']<(df_index['ABL_factor']*df_index['abl']) ), df_index['q_diff']/df_index['q'], 0)
            df_index = df_index.drop(columns=['ABL_factor']) 
        else: # use default ABL_factor
            df_index['f0'] = np.where( (df_index['q_diff']>default_q_diff_e) & ( df_index['z']<(default_abl_factor*df_index['abl']) ), df_index['q_diff']/df_index['q'], 0)       
        #循环更新计算权重f
        df_index['f'] = df_index['f0']
        for l in range(2,len(df_index)):            
            #df_index['f'].iloc[l:] = df_index['f'].iloc[l:]*(1-df_index['f0'].iloc[l-1])
            df_index.iloc[l:, df_index.columns.get_loc('f')] = df_index['f'].iloc[l:]*(1-df_index['f0'].iloc[l-1])        
        df_index = df_index.drop(columns=['f0'])  
        df.append(df_index)
        print(start_times[i],'establishing tracking table', n,'/',len(indexs))
        n = n+1
    writepkl(os.path.join(temporary_file_path, "{}-{}.pkl".format(start_times[i],'P particle tracking table') ),df)
print('temporary_file done!')

#%%