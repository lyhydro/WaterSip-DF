# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from main_functions import global_gridcell_info, get_files, readpkl, write_to_nc

#%% * * * * * * * * * * * * * * * * * * * * * setting * * * * * * * * * * * * * * * * * * * * * 
# start and end time of the temporary_file 
start_temporary_file_time = '2023071600' 
end_temporary_file_time = '2023073118' # end time minus time span
time_span = 6
output_spatial_resolution = 1
temporary_file_path = r'D:\Articles\WaterSip-DF\code-new\YRB\temporary'
final_output_path = r'D:\Articles\WaterSip-DF\code-new\YRB'

#%% * * * * * * * * * * * * * * * * * * * * * MAIN * * * * * * * * * * * * * * * * * * * * * 
latitude, longitude, gridcell_area = global_gridcell_info(output_spatial_resolution, lat_nor=90, lat_sou=-90, lon_lef=-179, lon_rig=180)
time = pd.date_range(start=pd.to_datetime(start_temporary_file_time, format='%Y%m%d%H'), end=pd.to_datetime(end_temporary_file_time, format='%Y%m%d%H'), freq='{}H'.format(time_span))
files = get_files(temporary_file_path, start_temporary_file_time, end_temporary_file_time, time_span, key_string='P particle tracking table')

moisture_contribution_kg = np.zeros([len(files), len(latitude), len(longitude)])
for n in range(0,len(files)):
    raw = readpkl(files[n])
    for i in range(0,len(raw)):
        part = raw[i]
        p_mass = part.iloc[0,part.columns.get_loc('p_mass')]
        for j in range(1,len(part)): # j=0 is P process
            lat_loc = np.argmin(np.abs(latitude - part.iloc[j,part.columns.get_loc('lat')]))
            lon_loc = np.argmin(np.abs(longitude - part.iloc[j,part.columns.get_loc('lon')]))
            moisture_contribution_kg[n,lat_loc,lon_loc] = moisture_contribution_kg[n,lat_loc,lon_loc] + p_mass*part.iloc[j,part.columns.get_loc('f')]
        print(files[n][-40:-30],"final step",i,'/',len(raw))
moisture_contribution_mm = moisture_contribution_kg/gridcell_area
print('final moisture source contribution (mm) done!')
#
write_to_nc(moisture_contribution_mm, name='moisture_contribution_mm', time=time, latitude=latitude, longitude=longitude, output_path=final_output_path)
