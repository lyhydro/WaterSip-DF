# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import xarray as xr
from main_functions import get_files, readpartposit_to_df, global_gridcell_info, midpoint, write_to_nc

#%% * * * * * * * * * * * * * * * * * * * * * setting * * * * * * * * * * * * * * * * * * * * * 
partposit_path = r'F:\FLEXPART\output_2023_July_EAg' # FLEXPART output
observation_file = r'D:\Articles\WaterSip-DF\data\gobal_TP_E_2023_June_July_1h.nc' # ERA5 hourly 'e' data

# recommend calculate one month at a time
start = '2023070100' # start time for P/E calculation
end = '2023080100' # end time for P/E calculation, include time_span
time_span = 6 # hours, time-step of partposit files
q_diff_e = 0.0000 # threshold used to extract all potential E particles
output_spatial_resolution = 1 # degree, spatial resolution for simulation output
output_path = r'D:\Articles\WaterSip-DF\code-new\temporary'

#%% * * * * * * * * * * * * * * * * * * * * * MAIN * * * * * * * * * * * * * * * * * * * * * 
time = pd.date_range(start=pd.to_datetime(start, format='%Y%m%d%H'), end=pd.to_datetime(end, format='%Y%m%d%H'), freq='{}H'.format(time_span))[:-1]
E_era5_6h = -xr.open_dataset( observation_file )['e'].resample(time='6H').sum(dim='time')*1000 # hourly to 6-hourly, m to mm

files = get_files(partposit_path, start, end, time_span)
latitude, longitude, gridcell_area = global_gridcell_info(output_spatial_resolution, lat_nor=90, lat_sou=-90, lon_lef=-179, lon_rig=180)

E_era5_6h = E_era5_6h.where(E_era5_6h > 0, 0) # make e<0 to 0
E_era5_6h = E_era5_6h.sel(time=time)*gridcell_area # mm to kg
DF_BLH_factors = np.zeros([len(time), np.shape(gridcell_area)[0], np.shape(gridcell_area)[1]])

for i in range(1,len(files)): # from 1 start, the files[0] used to calculate diff
    # read end-time partposit
    df = readpartposit_to_df(files[i], variables=['lon', 'lat', 'q', 'z', 'blh', 'mass'])
    # read one-time-step before partposit
    df0 = readpartposit_to_df(files[i-1], variables=['lon', 'lat', 'q', 'z', 'blh'])
    df0 = df0.rename(columns={'lat': 'lat0', 'lon': 'lon0', 'q': 'q0', 'z': 'z0', 'blh': 'blh0'})
    #
    df = df.merge(df0, left_index=True, right_index=True) 
    # q_diff threshold filter
    df['q_diff'] =  df['q'] - df['q0']
    df = df[df['q_diff'] > q_diff_e] 
    df = df.drop(columns=['q','q0'])
    # calculate middle-lat and middle-lon
    df['mid_lat'], df['mid_lon'] = midpoint(df['lat'].values, df['lon'].values, df['lat0'].values, df['lon0'].values)
    df = df.drop(columns=['lat','lon', 'lat0','lon0'])    
    # z and blh middle
    df['mid_z'] =  (df['z']+df['z0']) / 2
    df['mid_blh'] =  (df['blh']+df['blh0']) / 2
    df = df.drop(columns=['z','blh','z0','blh0'])    
    #
    df['lat_round'] = (df['mid_lat'] / output_spatial_resolution).round() * output_spatial_resolution
    df['lon_round'] = (df['mid_lon'] / output_spatial_resolution).round() * output_spatial_resolution
    df = df.drop(columns=['mid_lon','mid_lat'])
    # df = df[(df['lon_round'] <= 180) & (df['lon_round'] >= -179)] 
    # calculate the dynamic BLH scaling factors
    BLH_factors_span = np.arange(0.2,10.0,0.02) # set the smallest scaling factors step 0.02
    e_simulation = np.zeros([len(BLH_factors_span),len(latitude),len(longitude)]) 
    for j in range(0, len(BLH_factors_span)): 
        df0 = df.copy()
        df0 = df0[df0['mid_z'] < BLH_factors_span[j]*df0['mid_blh']]
        # calculate e_mass          
        df0['e_mass'] = df0['mass']*df0['q_diff'] # kg
        df_grouped_e = df0.groupby(['lat_round', 'lon_round'])['e_mass'].sum().reset_index()    
        # e写入0矩阵
        lat_idx_e = ((latitude[-1] - df_grouped_e['lat_round']) / output_spatial_resolution).astype(int)
        lon_idx_e = ((df_grouped_e['lon_round'] - longitude[0]) / output_spatial_resolution).astype(int)       
        e_simulation[j, lat_idx_e, lon_idx_e] = df_grouped_e['e_mass']
        print(f"{BLH_factors_span[j]:.2f}",'- ', end='')

    obs0 = E_era5_6h[i-1].values         
    min_indices = np.nanargmin(np.abs(e_simulation-obs0), axis=0)  
    # if all RH thresholds have no P, thus RH threshold is nan
    DF_BLH_factors[i-1] = np.where( np.all(e_simulation==0,axis=0), np.nan, BLH_factors_span[min_indices] )  
    DF_BLH_factors[i-1][obs0 == 0] = 0  # if no E, BLH_factors equals 0  
    print( files[i-1].split("partposit_")[-1], "done ！！！")

# store output data
write_to_nc(DF_BLH_factors, name='DF_BLH_factors', time=time, latitude=latitude, longitude=longitude, output_path=output_path)
