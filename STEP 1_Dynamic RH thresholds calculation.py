# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import xarray as xr
from main_functions import get_files, readpartposit_to_df, global_gridcell_info, calculate_RH, midpoint, write_to_nc

#%% * * * * * * * * * * * * * * * * * * * * * setting * * * * * * * * * * * * * * * * * * * * * 
partposit_path = r'F:\FLEXPART\output_2023_July_EAg' # FLEXPART output
observation_file = r'D:\Articles\WaterSip-DF\data\gobal_TP_E_2023_June_July_1h.nc' # ERA5 hourly 'tp' data

# recommend calculate one month at a time
start = '2023070100' # start time for P/E calculation
end = '2023080100' # end time for P/E calculation, include time_span
time_span = 6 # hours, time-step of partposit files
q_diff_p = -0.0000 # threshold used to extract all potential P particles
output_spatial_resolution = 1 # degree, spatial resolution for simulation output
output_path = r'D:\Articles\WaterSip-DF\code-new\temporary'

#%% * * * * * * * * * * * * * * * * * * * * * MAIN * * * * * * * * * * * * * * * * * * * * * 
time = pd.date_range(start=pd.to_datetime(start, format='%Y%m%d%H'), end=pd.to_datetime(end, format='%Y%m%d%H'), freq='{}H'.format(time_span))[:-1]
P_era5_6h = xr.open_dataset( observation_file )['tp'].resample(time='6H').sum(dim='time')*1000 # hourly to 6-hourly, m to mm

files = get_files(partposit_path, start, end, time_span)
latitude, longitude, gridcell_area = global_gridcell_info(output_spatial_resolution, lat_nor=90, lat_sou=-90, lon_lef=-179, lon_rig=180)

P_era5_6h = P_era5_6h.where(P_era5_6h > 0, 0) # make p<0 to 0
P_era5_6h = P_era5_6h.sel(time=time)*gridcell_area # mm to kg
DF_RH_thresholds = np.zeros([len(time), np.shape(gridcell_area)[0], np.shape(gridcell_area)[1]])

for i in range(1,len(files)): # from 1 start, the files[0] used to calculate diff
    # read end-time partposit
    df = readpartposit_to_df(files[i], variables=['lon', 'lat', 'q', 't', 'dens', 'mass'])
    calculate_RH(df)
    df = df.drop(columns=['t','dens'])
    # read one-time-step before partposit
    df0 = readpartposit_to_df(files[i-1], variables=['lon', 'lat', 'q', 't', 'dens']) 
    df0 = df0.rename(columns={'lat': 'lat0', 'lon': 'lon0', 'q': 'q0', 't': 't0', 'dens': 'dens0'})
    calculate_RH(df0, RH_name='RH0', dens='dens0', q='q0', t='t0')   
    df0 = df0.drop(columns=['t0','dens0'])
    #
    df = df.merge(df0, left_index=True, right_index=True) 
    # q_diff filtering
    df['q_diff'] =  df['q'] - df['q0']
    df = df[df['q_diff'] < q_diff_p] 
    df = df.drop(columns=['q','q0'])
    # calculate middle-lat and middle-lon
    df['mid_lat'], df['mid_lon'] = midpoint(df['lat'].values, df['lon'].values, df['lat0'].values, df['lon0'].values)
    df = df.drop(columns=['lat','lon', 'lat0','lon0'])
    # mid_RH
    df['mid_RH'] =  (df['RH']+df['RH0']) / 2
    df = df.drop(columns=['RH','RH0'])
    #
    df['lat_round'] = (df['mid_lat'] / output_spatial_resolution).round() * output_spatial_resolution
    df['lon_round'] = (df['mid_lon'] / output_spatial_resolution).round() * output_spatial_resolution
    df = df.drop(columns=['mid_lon','mid_lat'])
    # df = df[(df['lon_round'] <= 180) & (df['lon_round'] >= -179)]
    # calculate the dynamic RH thresholds
    RH_span = np.arange(0,100.1,0.2) # set the smallest RH thresholds step 0.2
    p_simulation = np.zeros([len(RH_span),len(latitude),len(longitude)]) 
    for j in range(0,len(RH_span)):
        df0 = df.copy()
        df0 = df0[ df0['mid_RH'] > RH_span[j] ]
        # calculate p_mass
        df0['p_mass'] = -df0['mass']*df0['q_diff'] #kg, *-1 makes P positive    
        # sum p on grids
        df_grouped_p = df0.groupby(['lat_round', 'lon_round'])['p_mass'].sum().reset_index()    
        # p to martix
        lat_idx_p = ((latitude[-1] - df_grouped_p['lat_round']) / output_spatial_resolution).astype(int)
        lon_idx_p = ((df_grouped_p['lon_round'] - longitude[0]) / output_spatial_resolution).astype(int) 
        p_simulation[j, lat_idx_p, lon_idx_p] = df_grouped_p['p_mass']
        print(f"{RH_span[j]:.1f}",'- ', end='')
    obs0 = P_era5_6h[i-1].values         
    min_indices = np.nanargmin(np.abs(p_simulation-obs0 ), axis=0)  
    # if all RH thresholds have no P, thus RH threshold is nan
    DF_RH_thresholds[i-1] = np.where( np.all(p_simulation==0,axis=0), np.nan, RH_span[min_indices] )  
    DF_RH_thresholds[i-1][obs0 == 0] = 100  # if no P, RH threshold equals 100  
    print( files[i-1].split("partposit_")[-1], "done ！！！")
    
# store output data
write_to_nc(DF_RH_thresholds, name='DF_RH_thresholds', time=time, latitude=latitude, longitude=longitude, output_path=output_path)

