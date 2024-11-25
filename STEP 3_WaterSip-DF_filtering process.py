# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
import xarray as xr
from main_functions import get_files, readpartposit_to_df, from_shp_get_parts, calculate_RH, midpoint, back_rtacking_files, writepkl

#%% * * * * * * * * * * * * * * * * * * * * * setting * * * * * * * * * * * * * * * * * * * * * 
flexpart_output_file = r'F:\FLEXPART\output_2023_July_EAg'
target_shp = r'D:\Articles\WaterSip-DF\code-new\YRB\boundary.shp'
start = '2023071600' # the diagnose start time
end = '2023080100' # the diagnose end time, include time-step
tracking_days = 15
output_spatial_resolution = 1
time_span = 6
temporary_file_path = r'D:\Articles\WaterSip-DF\code-new\YRB\temporary'
#
dynamic_RH_threshold = 'on'  # 'on' or 'off', 'on' for DF method
dynamic_BLH_factor = 'on'  # 'on' or 'off', 'on' for DF method
# DF setting
q_diff_p = -0.0000 # kg/kg
q_diff_e = 0.0000 # kg/kg
dynamic_rh_threshold_file = r'D:\Articles\WaterSip-DF\code-new\temporary\DF_RH_thresholds.nc'
dynamic_blh_factor_file = r'D:\Articles\WaterSip-DF\code-new\temporary\DF_BLH_factors.nc'
# default WaterSip setting
default_q_diff_p = -0.0002 # kg/kg
default_q_diff_e = 0.0002 # kg/kg
default_rh_threshold = 80 # %
default_abl_factor = 1.5 # BLH scailing factor

#%% * * * * * * * * * * * * * * * * * * * * * MAIN - found all P particles * * * * * * * * * * * * * * * * * * * * * 
files = get_files(flexpart_output_file, start, end, time_span)
# selecting all P particles in each time step in the shp defined target region
all_P_parts = []; times = []
for i in range(1,len(files)): # begain from 1 to calculate q_diff
    # read partposit
    df = readpartposit_to_df(files[i], variables=['lon', 'lat', 'q', 't', 'dens', 'mass'])
    calculate_RH(df)
    # read one-time-step before partposit
    df0 = readpartposit_to_df(files[i-1], variables=['lon', 'lat', 'q', 't', 'dens'])
    df0 = df0.rename(columns={'lat': 'lat0', 'lon': 'lon0', 'q': 'q0', 't': 't0', 'dens': 'dens0'})   
    calculate_RH(df0, RH_name='RH0', dens='dens0', q='q0', t='t0')
    # merge
    df = df.merge(df0, left_index=True, right_index=True) 
    df = df.drop(columns=['dens','dens0', 't', 't0'])
    # q_diff threshold filter
    df['q_diff'] =  df['q'] - df['q0']
    df = df.drop(columns=['q','q0'])
    df = df[df['q_diff'] < q_diff_p] 
    # middle lat and lon
    df['lat'], df['lon'] = midpoint(df['lat'].values, df['lon'].values, df['lat0'].values, df['lon0'].values)
    df = df.drop(columns=['lat0','lon0'])
    # 筛选研究区粒子
    df = from_shp_get_parts(df,target_shp)
    # mid_RH
    df['mid_RH'] =  (df['RH']+df['RH0']) / 2
    df = df.drop(columns=['RH','RH0'])   
    ## first correct by rh_threshold
    if dynamic_RH_threshold == 'on':
        df['lat_round'] = (df['lat'] / output_spatial_resolution).round() * output_spatial_resolution
        df['lon_round'] = (df['lon'] / output_spatial_resolution).round() * output_spatial_resolution  
        # df = df[(df['lon_round'] <= 180) & (df['lon_round'] >= -179)] # obs coverage
        time_dt = pd.to_datetime(files[i-1][-14:-4], format='%Y%m%d%H') 
        DF_RH_thresholds = xr.open_dataset( dynamic_rh_threshold_file )[ 'DF_RH_thresholds' ]
        RH_threshold = DF_RH_thresholds.sel(time=time_dt)
        df_RH_threshold = RH_threshold.to_dataframe(name='RH_threshold').reset_index().drop(columns=['time'])
        original_index = df.index # keep index
        df = pd.merge(df, df_RH_threshold, how='left', left_on=['lat_round', 'lon_round'], right_on=['latitude', 'longitude'])  
        df.index = original_index # recover index
        df.dropna(subset=['RH_threshold'], inplace=True) # drop nan
        df = df[df['RH_threshold'] != 100] # drop rh_threshold=100
        df = df[ df['mid_RH'] >= df['RH_threshold'] ]   
        df = df.drop(columns=['latitude', 'longitude', 'mid_RH', 'RH_threshold', 'lat_round', 'lon_round'])             
    else: # use default rh_threshold
        df = df[df['q_diff'] < default_q_diff_p]
        df = df[df['mid_RH'] > default_rh_threshold]
    # calculate p simulation
    df['p_mass'] =  -df['mass']*df['q_diff']
    all_P_parts.append(df)
    times.append(files[i-1][-14:-4])
    if len(df)==0:
        print("!!!!! no parts found at this time-step !!!!!")
    print('selecting P particles in',files[i-1][-14:-1])

#%% * * * * * * * * * * * * * * * * * * * * * MAIN - tracking processes * * * * * * * * * * * * * * * * * * * * * 
DF_BLH_factors = xr.open_dataset( dynamic_blh_factor_file )[ 'DF_BLH_factors' ]
df_BLH_factor = DF_BLH_factors.to_dataframe(name='BLH_factor').reset_index()
# get and store P particles backtrack files in each time-step, and establish tracking table for each particle
for i in range(0,len(times)):
    result = []
    files_track = back_rtacking_files(flexpart_output_file, times[i], tracking_days, time_span)
    result.append(all_P_parts[i]) 
    for f in files_track[::-1]:
        df0 = readpartposit_to_df(f, variables=['lon', 'lat', 'z', 'q', 'blh', 'mass'])
        result.append( df0.loc[df0.index.intersection(result[-1].index)] )
        print('.',end='')
    writepkl("{}\{}-{} P particle backward files.pkl".format(temporary_file_path,times[i],time_span),result)
    print('\n',times[i],'P particles get backtrack files done')
    
    ## establish particle tracking table
    indexs = result[0].index 
    df0 = []; n = 1
    track_times = pd.to_datetime(times[i], format='%Y%m%d%H') - pd.Timedelta(hours=time_span) * np.arange(len(result)-1)
    for index in indexs:
        # each particle has a dataframe table from tracking start to end
        df_index = []
        df_index.append( result[0].loc[index][['lon','lat','q_diff','mass','p_mass']] )
        for j in range(1,len(result)):
            if index in result[j].index:
                df_index.append( result[j].loc[index][['lon','lat','z','q','blh']] ) 
            else:
                df_index.append( df_index[-1].apply(lambda x: np.nan) ) # if particle lost, set nan
        df_index = pd.DataFrame(df_index) 
        # lat_mid, lon_mid, z_mid, q_diff, and abl_mid
        df_index['lat'][1:-1], df_index['lon'][1:-1] = midpoint(df_index['lat'][1:-1].values, df_index['lon'][1:-1].values, df_index['lat'][2:].values, df_index['lon'][2:].values)
        df_index['z'][1:-1] = (df_index['z'][1:-1].values + df_index['z'][2:].values) / 2
        df_index['q_diff'][1:-1] = df_index['q'][1:-1].values - df_index['q'][2:].values
        df_index['blh'][1:-1] = (df_index['blh'][1:-1].values + df_index['blh'][2:].values) / 2
        df_index = df_index.iloc[:-1]
        # 
        if dynamic_BLH_factor == 'on': # use dynamic ABL_factor
            df_index['lat_round'] = (df_index['lat'] / output_spatial_resolution).round() * output_spatial_resolution
            df_index['lon_round'] = (df_index['lon'] / output_spatial_resolution).round() * output_spatial_resolution 
            df_index['start_time'] = track_times
            df_index = pd.merge(df_index, df_BLH_factor, how='left', left_on=['lat_round', 'lon_round', 'start_time'], right_on=['latitude', 'longitude', 'time'])         
            df_index = df_index.drop(columns=['latitude', 'longitude', 'start_time', 'time','lat_round', 'lon_round'])    
            # selecting E region, and calculate contribution ratio f in each time step
            df_index['f0'] = np.where( (df_index['q_diff']>q_diff_e) & ( df_index['z']<(df_index['BLH_factor']*df_index['blh']) ), df_index['q_diff']/df_index['q'], 0)
            df_index = df_index.drop(columns=['BLH_factor']) 
        else: # use default ABL_factor
            df_index['f0'] = np.where( (df_index['q_diff']>default_q_diff_e) & ( df_index['z']<(default_abl_factor*df_index['blh']) ), df_index['q_diff']/df_index['q'], 0)       
        df_index = df_index.drop(columns=['z','blh','q_diff','q']) 
        # cycle to weight f0 to f
        df_index['f'] = df_index['f0']
        for l in range(2,len(df_index)):  
            #df_index['f'].iloc[l:] = df_index['f'].iloc[l:]*(1-df_index['f0'].iloc[l-1])
            df_index.iloc[l:, df_index.columns.get_loc('f')] = df_index['f'].iloc[l:]*(1-df_index['f0'].iloc[l-1])        
        df_index = df_index.drop(columns=['f0'])  
        df0.append(df_index)
        print(times[i],'establishing tracking table', n,'/',len(indexs))
        n = n+1
    writepkl(os.path.join(temporary_file_path, "{}-{} {}.pkl".format(times[i], time_span,'P particle tracking table') ),df0)
print('temporary_file done!')

#%%