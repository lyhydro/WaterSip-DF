# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from datetime import timedelta
from functions import (get_files_temporary, readpkl, global_gridcell_info, write_to_nc_3d)
import config


def calculate_moisture_contribution(df_method):

    print('\nStep 4: Calculate moisture contribution...')
    
    # Note that FLEXPART10.4 output lon are from -179 to 180 !8
    latitude, longitude, gridcell_area = global_gridcell_info(config.OUTPUT_SPATIAL_RESOLUTION, 
                                                              lat_nor=90, lat_sou=-90,
                                                              lon_lef=-179, lon_rig=180)
    
    start = pd.to_datetime(config.START_TIME, format='%Y%m%d%H')
    end = pd.to_datetime(config.END_TIME, format='%Y%m%d%H') - timedelta(hours=config.TIME_SPAN)
    time = pd.date_range(start=start, end=end, freq='{}h'.format(config.TIME_SPAN))
    
    files = get_files_temporary(config.TEMPORARY_FILE_PATH, config.START_TIME, 
                               end.strftime('%Y%m%d%H'), config.TIME_SPAN)
    
    moisture_contribution_kg = np.zeros([len(files), len(latitude), len(longitude)])
    
    for n in range(0, len(files)):
        raw = readpkl(files[n])
        
        for i in range(0, len(raw)):
            part = raw[i]
            
            if part is None or len(part) == 0:
                continue
            
            p_mass = part.iloc[0, part.columns.get_loc('p_mass')]
            
            for j in range(1, len(part)):
                lat = part.iloc[j, part.columns.get_loc('lat')]
                lon = part.iloc[j, part.columns.get_loc('lon')]
                f_value = part.iloc[j, part.columns.get_loc('f')]
                
                if pd.isna(lat) or pd.isna(lon) or pd.isna(f_value) or f_value==0:
                    continue
                
                lat_loc = np.argmin(np.abs(latitude - lat))
                lon_loc = np.argmin(np.abs(longitude - lon))
                
                moisture_contribution_kg[n, lat_loc, lon_loc] += p_mass * f_value
        print('.', end='')
    
    moisture_contribution_mm = moisture_contribution_kg / gridcell_area
    
    if df_method == 'on':
        output_name_mm = 'moisture_contribution_mm_DF_{}_{}'.format(config.START_TIME, config.END_TIME)
        output_name_kg = 'moisture_contribution_kg_DF_{}_{}'.format(config.START_TIME, config.END_TIME)
    else:
        output_name_mm = 'moisture_contribution_mm_{}_{}'.format(config.START_TIME, config.END_TIME)
        output_name_kg = 'moisture_contribution_kg_{}_{}'.format(config.START_TIME, config.END_TIME)
    
    write_to_nc_3d(moisture_contribution_mm, 'moisture_contribution_mm', output_name_mm, 
                   time, latitude, longitude, config.FINAL_OUTPUT_PATH)
    
    write_to_nc_3d(moisture_contribution_kg, 'moisture_contribution_kg', output_name_kg, 
                   time, latitude, longitude, config.FINAL_OUTPUT_PATH)
    
    print('Step 4 done!')


if __name__ == "__main__":
    calculate_moisture_contribution('on') 