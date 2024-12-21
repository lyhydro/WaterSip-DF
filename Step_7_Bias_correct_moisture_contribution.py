# -*- coding: utf-8 -*-
import numpy as np
import xarray as xr
import pandas as pd
from main_functions import get_and_combine_obs_files, global_gridcell_info, from_shp_get_mask, from_bounds_get_mask, write_to_nc_2d
from YAMLConfig import YAMLConfig


#%% * * * * * * * * * * MAIN * * * * * * * * * *
def bias_correct_moisture_contribution(df_method):
    config = YAMLConfig('config.yaml')
    general_config = config.get('general')
    tracking_days = general_config['tracking_days']
    start_time = str(general_config['start_time'])
    end_time = str(general_config['end_time'])

    path_config = config.get('path')
    observation_path = config.get('warerSip-DF-HAMSTER')['observation_path']
    final_output_path = general_config['final_output_path']
    target_region = general_config['target_region']
    P_E_simulation_output_path = config.get('warerSip-DF-HAMSTER')['P_E_simulation_output_path']

    if df_method == 'on':
        moisture_contribution_mm = xr.open_dataset('{}\moisture_contribution_mm_DF_{}_{}.nc'.format(final_output_path, start_time, end_time))['moisture_contribution_mm']
    else:
        moisture_contribution_mm = xr.open_dataset('{}\moisture_contribution_mm_{}_{}.nc'.format(final_output_path, start_time, end_time))['moisture_contribution_mm']

    time_p = pd.to_datetime(moisture_contribution_mm['time'].values)
    time_e = pd.date_range(time_p[0] - pd.Timedelta(days=tracking_days), time_p[-1], freq='6H')

    #%% ******************** e correct coeffcience ********************
    ds = get_and_combine_obs_files(observation_path, time_e.strftime('%Y%m').drop_duplicates(), variable='e')
    obs_e = -ds.resample(time='6H').sum(dim='time')*1000
    obs_e = obs_e.sel(time=time_e)
    obs_e = obs_e.where(obs_e > 0, 0) # 粒子动态筛选是在6小时尺度上，将所有小于0的值都置零了，这里保持一致
    simu_global_e = []
    for t in time_e:
        simu_global_e.append(xr.open_dataset('{}\E_simulation_mm_{}.nc'.format(P_E_simulation_output_path,t.strftime('%Y%m%d%H')))['E_simulation_mm'] )
    simu_global_e = xr.concat(simu_global_e, dim="time")
    simu_global_e = simu_global_e.where(simu_global_e > 0, 0)
    #在整个水汽追踪时段进行了校正；如果在每一个time_step进行校正，校正系数会存在无穷大值和无穷小值！！！
    cc_e = obs_e.sum(dim='time') / simu_global_e.sum(dim='time')
    print('e correct coeffcience done')

    #%% ******************** p correct coeffcience ********************
    latitude, longitude, gridcell_area = global_gridcell_info(1, lat_nor=90, lat_sou=-90, lon_lef=-179, lon_rig=180)
    if target_region[-4:]=='.shp':
        mask = from_shp_get_mask(target_region, latitude, longitude)
    elif isinstance(target_region, list):
        mask = from_bounds_get_mask(target_region[0], target_region[1], target_region[2], target_region[3], latitude, longitude)
    ds = get_and_combine_obs_files(observation_path, time_p.strftime('%Y%m').drop_duplicates(), variable='tp')
    obs_p = ds.resample(time='6H').sum(dim='time')*1000
    obs_p = obs_p.sel(time=time_p)
    obs_p = obs_p.where(obs_p > 0, 0)
    simu_global_p = []
    for t in time_p:
        simu_global_p.append(xr.open_dataset('{}\P_simulation_mm_{}.nc'.format(P_E_simulation_output_path, t.strftime('%Y%m%d%H')) )['P_simulation_mm'] )
    simu_global_p = xr.concat(simu_global_p, dim="time")
    simu_global_p = simu_global_p.where(simu_global_p > 0, 0)

    # ******************** correct ********************
    moisture_contribution_mm_sum = moisture_contribution_mm.sum(dim='time')
    cc_e = cc_e.where(cc_e <= 10, other=10)
    cc_e = cc_e.where(cc_e > 0, other=0)
    cc_p = np.sum(obs_p*gridcell_area*mask) * ( np.sum(moisture_contribution_mm_sum*gridcell_area)/np.sum(simu_global_p*gridcell_area*mask) ) / np.sum(moisture_contribution_mm_sum*gridcell_area*cc_e)
    print('p correct coeffcience', cc_p.values, 'done')

    #%%
    moisture_contribution_mm_sum_corrected = moisture_contribution_mm_sum*cc_e*cc_p
    if df_method == 'on':
        write_to_nc_2d(moisture_contribution_mm_sum_corrected, 'moisture_contribution_mm'
                       ,'moisture_contribution_mm_DF_HAMSTER_{}_{}'.format(start_time, end_time)
                       ,latitude, longitude, final_output_path)
    else:
        write_to_nc_2d(moisture_contribution_mm_sum_corrected, 'moisture_contribution_mm'
                       ,'moisture_contribution_mm_HAMSTER_{}_{}'.format(start_time, end_time)
                       ,latitude, longitude, final_output_path)
