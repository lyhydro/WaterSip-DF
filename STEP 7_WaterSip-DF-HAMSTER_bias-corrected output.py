# -*- coding: utf-8 -*-
import numpy as np
import xarray as xr
import pandas as pd
from main_functions import global_gridcell_info, write_to_nc, from_shp_get_mask

obs_file = r'D:\Articles\An optimized bias-correction framework for moisture source-receptor diagnose in Lagrangian models\data\gobal_TP_E_2023_June_July_1h.nc'

simu_file_global_p = r'D:\Articles\An optimized bias-correction framework for moisture source-receptor diagnose in Lagrangian models\data\P_simulation_mm_by_RH_threshold_opt.nc'
simu_file_global_e = r'D:\Articles\An optimized bias-correction framework for moisture source-receptor diagnose in Lagrangian models\data\E_simulation_mm_by_ABL_factor_opt.nc'

moisture_contribution_file = r'D:\Articles\An optimized bias-correction framework for moisture source-receptor diagnose in Lagrangian models\YRB\final_output\moisture_contribution_mm_DF.nc'
shp_file = r'D:\Articles\An optimized bias-correction framework for moisture source-receptor diagnose in Lagrangian models\YRB\shp\boundary.shp'

tracking_days = 15

output_path = r'D:\Articles\An optimized bias-correction framework for moisture source-receptor diagnose in Lagrangian models\YRB\final_output'

#%%
moisture_contribution_mm = xr.open_dataset(moisture_contribution_file)['moisture_contribution_mm']
time_p = moisture_contribution_mm['time']
time_e = pd.date_range(time_p[0].values - pd.Timedelta(days=tracking_days), time_p[-1].values, freq='6H')

#%% ******************** e correct coeffcience ********************
obs_e = -xr.open_dataset( obs_file )['e'].resample(time='6H').sum(dim='time')*1000
obs_e = obs_e.sel( time=time_e)
obs_e = obs_e.where(obs_e > 0, 0) # 粒子动态筛选是在6小时尺度上，将所有小于0的值都置零了，这里保持一致
simu_global_e = xr.open_dataset( simu_file_global_e )[ list(xr.open_dataset(simu_file_global_e).data_vars.keys())[0] ]
simu_global_e = simu_global_e.sel( time=time_e)
simu_global_e = simu_global_e.where(simu_global_e > 0, 0)
# cc = obs/simu #在每一个time_step进行校正，校正系数会存在无穷大值和无穷小值！！！
#本研究仅在月尺度的格点上进行了校正，如下
cc_e = obs_e.sum(dim='time') / simu_global_e.sum(dim='time')
print('e correct coeffcience done')

#%% ******************** p correct coeffcience ********************
latitude, longitude, gridcell_area = global_gridcell_info(1, lat_nor=90, lat_sou=-90, lon_lef=-179, lon_rig=180)
mask = from_shp_get_mask(shp_file, latitude, longitude)
obs_p = xr.open_dataset( obs_file )['tp'].resample(time='6H').sum(dim='time')*1000
obs_p = obs_p.sel( time=time_p)
obs_p = obs_p.where(obs_p > 0, 0) 
simu_global_p = xr.open_dataset( simu_file_global_p )[ list(xr.open_dataset(simu_file_global_p).data_vars.keys())[0] ]
simu_global_p = simu_global_p.sel( time=time_p)
simu_global_p = simu_global_p.where(simu_global_p > 0, 0)

# ******************** correct ********************
moisture_contribution_mm_sum = moisture_contribution_mm.sum(dim='time')
cc_e = cc_e.where(cc_e <= 10, other=10)
cc_e = cc_e.where(cc_e > 0, other=0)
cc_p = np.sum(obs_p*gridcell_area*mask) * ( np.sum(moisture_contribution_mm_sum*gridcell_area)/np.sum(simu_global_p*gridcell_area*mask) ) / np.sum(moisture_contribution_mm_sum*gridcell_area*cc_e)
print('p correct coeffcience', cc_p.values, 'done')

#%%
moisture_contribution_mm_DF_sum_hamster = moisture_contribution_mm_sum*cc_e*cc_p
write_to_nc(moisture_contribution_mm_DF_sum_hamster, name='moisture_contribution_mm_DF_sum_hamster', time=0, latitude=latitude, longitude=longitude, output_path=output_path)
