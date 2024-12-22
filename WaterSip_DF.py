# -*- coding: utf-8 -*-
from Step_1_Dynamic_RH_thresholds_calculation import dynamic_rh_thresholds_calculation
from Step_2_Dynamic_BLH_factors_calculation import dynamic_blh_factors_calculation
from Step_3_Filtering_and_tracking import filtering_and_tracking
from Step_4_Calculate_moisture_contribution import calculate_moisture_contribution
from main_functions import check_df_file
from YAMLConfig import YAMLConfig


def main_watersip_df():
    config = YAMLConfig('config.yaml')
    update_DF = config.get('WaterSip-DF')['update_DF']
    # 检查DF文件夹是否有相应时刻的文件，有的话询问是否更新，选否的话，跳过step1和2
    result = check_df_file()
    if not result or update_DF:
        # step 1
        dynamic_rh_thresholds_calculation()
        print('step 1 done!')
        # step 2
        dynamic_blh_factors_calculation()
        print('step 2 done!')
    # step 3
    filtering_and_tracking('on')
    # step 4
    calculate_moisture_contribution('on')


if __name__ == "__main__":
    main_watersip_df()
    # filtering_and_tracking('on')
