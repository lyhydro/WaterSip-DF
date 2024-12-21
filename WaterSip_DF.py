# -*- coding: utf-8 -*-
from Step_1_Dynamic_RH_thresholds_calculation import dynamic_rh_thresholds_calculation
from Step_2_Dynamic_BLH_factors_calculation import dynamic_blh_factors_calculation
from Step_3_Filtering_and_tracking import filtering_and_tracking
from Step_4_Calculate_moisture_contribution import calculate_moisture_contribution



def main_watersip_df():
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
    filtering_and_tracking('on')
