# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from datetime import timedelta
from main_functions import global_gridcell_info, get_files, readpkl, write_to_nc_3d
from YAMLConfig import YAMLConfig


#%% * * * * * * * * * * * * * * * * * * * * * MAIN * * * * * * * * * * * * * * * * * * * * *
def calculate_moisture_contribution(df_method):
    print('begin step 4:moisture contribution !')
    config = YAMLConfig('config.yaml')
    general_config = config.get('general')
    time_span = general_config['time_span']
    output_spatial_resolution = general_config['output_spatial_resolution']
    start_time = str(general_config['start_time'])
    end_time = str(general_config['end_time'])

    partposit_path = general_config['partposit_path']
    DF_file_path = config.get('warerSip-DF')['DF_file_path']
    target_region = general_config['target_region']
    temporary_file_path = general_config['temporary_file_path']
    final_output_path = general_config['final_output_path']


    latitude, longitude, gridcell_area = global_gridcell_info(output_spatial_resolution, lat_nor=90, lat_sou=-90, lon_lef=-179, lon_rig=180)
    start = pd.to_datetime(start_time, format='%Y%m%d%H')
    end = pd.to_datetime(end_time, format='%Y%m%d%H')-timedelta(hours=6)
    time = pd.date_range(start=start, end=end, freq='{}H'.format(time_span))
    files = get_files(temporary_file_path, start_time, end.strftime('%Y%m%d%H'), time_span, key_string='P particle tracking table')
    #
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
    if df_method == 'on': # 'on' or 'off'
        write_to_nc_3d(moisture_contribution_mm, 'moisture_contribution_mm'
                       ,'moisture_contribution_mm_DF_{}_{}'.format(start_time,end_time)
                       ,time, latitude, longitude, final_output_path)
    else:
        write_to_nc_3d(moisture_contribution_mm, 'moisture_contribution_mm'
                       ,'moisture_contribution_mm_{}_{}'.format(start_time,end_time)
                       ,time, latitude, longitude, final_output_path)
    print('step 4 done!')