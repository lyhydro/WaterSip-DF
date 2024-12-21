# -*- coding: utf-8 -*-
from Step_1_Dynamic_RH_thresholds_calculation import dynamic_rh_thresholds_calculation
from Step_2_Dynamic_BLH_factors_calculation import dynamic_blh_factors_calculation
from Step_3_Filtering_and_tracking import filtering_and_tracking
from Step_4_Calculate_moisture_contribution import calculate_moisture_contribution
from Step_5_P_simulation import p_simulation
from Step_6_E_simulation import e_simulation
from Step_7_Bias_correct_moisture_contribution import bias_correct_moisture_contribution


def main_watersip_df_hamster():
    # step 1
    dynamic_rh_thresholds_calculation()
    # step 2
    dynamic_blh_factors_calculation()
    # step 3
    filtering_and_tracking('on')
    # step 4
    calculate_moisture_contribution('on')
    # step 5
    p_simulation('on')
    # step 6
    e_simulation('on')
    # step 7
    bias_correct_moisture_contribution('on')
