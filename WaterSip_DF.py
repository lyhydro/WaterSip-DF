# -*- coding: utf-8 -*-

from Step_1_Dynamic_RH_thresholds_calculation import dynamic_rh_thresholds_calculation
from Step_2_Dynamic_BLH_factors_calculation import dynamic_blh_factors_calculation
from Step_3_Filtering_and_tracking import filtering_and_tracking
from Step_4_Calculate_moisture_contribution import calculate_moisture_contribution
from functions import get_algorithm_params


def main_watersip_df():

    print("Starting WaterSip-DF ...")
    
    method = 'WaterSip-DF'
    params = get_algorithm_params(method)
    
    update_DF = params['update_df']

#---------------------------------------------------------------------------   
    
    if update_DF:
        # Step 1: Dynamic RH thresholds calculation
        dynamic_rh_thresholds_calculation(method)
        
        # Step 2: Dynamic BLH factors calculation
        dynamic_blh_factors_calculation(method)
    
    # Step 3: Filtering and tracking
    filtering_and_tracking(method, 'on')
    
    # Step 4: Calculate moisture contribution
    calculate_moisture_contribution('on')
    
    print("WaterSip-DF v3 completed!")


if __name__ == "__main__":
    main_watersip_df()