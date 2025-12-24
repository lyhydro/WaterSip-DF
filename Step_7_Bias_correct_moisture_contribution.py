# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import xarray as xr
import os
from functions import (get_and_combine_obs_files, global_gridcell_info, 
                      write_to_nc_2d, from_shp_get_mask, from_bounds_get_mask)
import config


def bias_correct_moisture_contribution(method, df_method):

    print(f'Step 7 ({method}): Bias correct moisture contribution...')
    
    if df_method == 'on':
        nc_file = f'{config.FINAL_OUTPUT_PATH}/moisture_contribution_mm_DF_{config.START_TIME}_{config.END_TIME}.nc'
    else:
        nc_file = f'{config.FINAL_OUTPUT_PATH}/moisture_contribution_mm_{config.START_TIME}_{config.END_TIME}.nc'
    
    if not os.path.exists(nc_file):
        raise FileNotFoundError(f"Moisture contribution file not found: {nc_file}")
    
    moisture_contribution_mm = xr.open_dataset(nc_file)['moisture_contribution_mm']
    
    time_p = pd.to_datetime(moisture_contribution_mm['time'].values)
    time_e = pd.date_range(time_p[0] - pd.Timedelta(days=config.TRACKING_DAYS), time_p[-1], freq='6h')
    
    # ******************** e correct coeffcience ********************
    print("Loading evaporation data")
    ds = get_and_combine_obs_files(config.OBSERVATION_PATH, time_e.strftime('%Y%m').drop_duplicates(), variable='e')
    obs_e = -ds.resample(time='6h').sum(dim='time') * 1000
    obs_e = obs_e.sel(time=time_e)
    obs_e = obs_e.where(obs_e > 0, 0)  
    
    e_files = [f'{config.P_E_SIMULATION_OUTPUT_PATH}/E_simulation_mm_{t.strftime("%Y%m%d%H")}.nc' for t in time_e]
    missing_files = [f for f in e_files if not os.path.exists(f)]
    if missing_files:
        raise FileNotFoundError(f"Missing E simulation files: {missing_files[:3]}...")
    
    simu_global_e = xr.open_mfdataset(e_files, combine='nested', concat_dim='time')['E_simulation_mm'] 
    simu_global_e = simu_global_e.where(simu_global_e > 0, 0)
    
    cc_e = obs_e.sum(dim='time') / simu_global_e.sum(dim='time')
    
    # ******************** p correct coeffcience ********************
    print("Loading precipitation data")
    latitude, longitude, gridcell_area = global_gridcell_info(config.OUTPUT_SPATIAL_RESOLUTION, 
                                                              lat_nor=90, lat_sou=-90,
                                                              lon_lef=-179, lon_rig=180)
    
    if isinstance(config.TARGET_REGION, str) and config.TARGET_REGION.endswith('.shp'):
        mask = from_shp_get_mask(config.TARGET_REGION, latitude, longitude)
    elif isinstance(config.TARGET_REGION, list):
        mask = from_bounds_get_mask(config.TARGET_REGION[0], config.TARGET_REGION[1], 
                                   config.TARGET_REGION[2], config.TARGET_REGION[3], latitude, longitude)
    else:

        mask = np.ones((len(latitude), len(longitude)))
    
    ds = get_and_combine_obs_files(config.OBSERVATION_PATH, time_p.strftime('%Y%m').drop_duplicates(), variable='tp')
    obs_p = ds.resample(time='6h').sum(dim='time') * 1000
    obs_p = obs_p.sel(time=time_p)
    obs_p = obs_p.where(obs_p > 0, 0)
    
    p_files = [f'{config.P_E_SIMULATION_OUTPUT_PATH}/P_simulation_mm_{t.strftime("%Y%m%d%H")}.nc' for t in time_p]
    missing_files = [f for f in p_files if not os.path.exists(f)]
    if missing_files:
        raise FileNotFoundError(f"Missing P simulation files: {missing_files[:3]}...")
    
    simu_global_p = xr.open_mfdataset(p_files, combine='nested', concat_dim='time')['P_simulation_mm'] 
    simu_global_p = simu_global_p.where(simu_global_p > 0, 0)
    
    # ******************** correct ********************
    print("Start correcting")
    moisture_contribution_mm_sum = moisture_contribution_mm.sum(dim='time')

    cc_e = cc_e.where(cc_e <= 10, other=10) 
    cc_e = cc_e.where(cc_e > 0, other=0)
    print('E correct coeffcience (global median):', cc_e.load().median().values)
    cc_p = np.sum(obs_p * gridcell_area * mask) * (np.sum(moisture_contribution_mm_sum * gridcell_area) / np.sum(simu_global_p * gridcell_area * mask)) / np.sum(moisture_contribution_mm_sum * gridcell_area * cc_e)
    print('P correct coeffcience:', cc_p.values)
    
    moisture_contribution_mm_sum_corrected = moisture_contribution_mm_sum * cc_e * cc_p
    
    print("Writing corrected results...")
    if df_method == 'on':
        write_to_nc_2d(moisture_contribution_mm_sum_corrected, 'moisture_contribution_mm',
                       f'moisture_contribution_mm_DF_HAMSTER_{config.START_TIME}_{config.END_TIME}',
                       latitude, longitude, config.FINAL_OUTPUT_PATH)
    else:
        write_to_nc_2d(moisture_contribution_mm_sum_corrected, 'moisture_contribution_mm',
                       f'moisture_contribution_mm_HAMSTER_{config.START_TIME}_{config.END_TIME}',
                       latitude, longitude, config.FINAL_OUTPUT_PATH)
    
    print('Step 7 done!')


if __name__ == "__main__":
    bias_correct_moisture_contribution('WaterSip-DF-HAMSTER', 'on') 