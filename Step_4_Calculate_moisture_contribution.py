# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from datetime import timedelta
from main_functions import global_gridcell_info, get_files_temporary, readpkl, write_to_nc_3d
from YAMLConfig import YAMLConfig
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class CalculateMoistureContribution:

    def __init__(self):
        config = YAMLConfig('config.yaml')
        general_config = config.get('General')
        self.time_span = general_config['time_span']
        self.output_spatial_resolution = general_config['output_spatial_resolution']
        self.start_time = str(general_config['start_time'])
        self.end_time = str(general_config['end_time'])
        self.temporary_file_path = general_config['temporary_file_path']
        self.final_output_path = general_config['final_output_path']

    # %% * * * * * * * * * * * * * * * * * * * * * MAIN * * * * * * * * * * * * * * * * * * * * *
    def calculate_moisture_contribution(self, df_method):
        logging.info('Step 4: Calculate moisture contribution...')

        latitude, longitude, gridcell_area = global_gridcell_info(self.output_spatial_resolution, lat_nor=90,
                                                                  lat_sou=-90,lon_lef=-179, lon_rig=180 )
        start = pd.to_datetime(self.start_time, format='%Y%m%d%H')
        end = pd.to_datetime(self.end_time, format='%Y%m%d%H') - timedelta(hours=self.time_span)
        time = pd.date_range(start=start, end=end, freq='{}h'.format(self.time_span))
        # files = get_files(self.temporary_file_path, self.start_time, end.strftime('%Y%m%d%H'), self.time_span,key_string='P particle tracking table')
        files = get_files_temporary(self.temporary_file_path, self.start_time, end.strftime('%Y%m%d%H'), self.time_span)
        #
        moisture_contribution_kg = np.zeros([len(files), len(latitude), len(longitude)])
        for n in range(0, len(files)):
            try:
                raw = readpkl(files[n])
                for i in range(0, len(raw)):
                    part = raw[i]
                    p_mass = part.iloc[0, part.columns.get_loc('p_mass')]
                    for j in range(1, len(part)):  # j=0 is P process
                        lat_loc = np.argmin(np.abs(latitude - part.iloc[j, part.columns.get_loc('lat')]))
                        lon_loc = np.argmin(np.abs(longitude - part.iloc[j, part.columns.get_loc('lon')]))
                        moisture_contribution_kg[n, lat_loc, lon_loc] = moisture_contribution_kg[
                                                                            n, lat_loc, lon_loc] + p_mass * \
                                                                            part.iloc[j, part.columns.get_loc('f')]
                logging.info(f'Fnish accumulation for {files[n][-42:-32]}, total {len(raw)} particles')
            except FileNotFoundError:
                print(f"File not found: {files[n]}, skipping to next file.")
                continue
            except Exception as e:
                print(f"An error occurred while processing file {files[n]}: {e}")
                continue
        moisture_contribution_mm = moisture_contribution_kg / gridcell_area

        #
        if df_method == 'on':  # 'on' or 'off'
            write_to_nc_3d(moisture_contribution_mm, 'moisture_contribution_mm'
                           , 'moisture_contribution_mm_DF_{}_{}'.format(self.start_time, self.end_time)
                           , time, latitude, longitude, self.final_output_path)
        else:
            write_to_nc_3d(moisture_contribution_mm, 'moisture_contribution_mm'
                           , 'moisture_contribution_mm_{}_{}'.format(self.start_time, self.end_time)
                           , time, latitude, longitude, self.final_output_path)
        logging.info('Step 4 done!')


if __name__ == "__main__":
    ft = CalculateMoistureContribution()
    ft.calculate_moisture_contribution('on')