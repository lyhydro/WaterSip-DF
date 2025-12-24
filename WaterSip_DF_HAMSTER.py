# -*- coding: utf-8 -*-

from Step_1_Dynamic_RH_thresholds_calculation import dynamic_rh_thresholds_calculation
from Step_2_Dynamic_BLH_factors_calculation import dynamic_blh_factors_calculation
from Step_3_Filtering_and_tracking import filtering_and_tracking
from Step_4_Calculate_moisture_contribution import calculate_moisture_contribution
from Step_5_P_simulation import p_simulation
from Step_6_E_simulation import e_simulation
from Step_7_Bias_correct_moisture_contribution import bias_correct_moisture_contribution
from functions import get_algorithm_params
import time


def main_watersip_df_hamster():

    print("Starting WaterSip-DF-HAMSTER...")
    
    method = 'WaterSip-DF-HAMSTER'
    params = get_algorithm_params(method)
    
    update_DF = params['update_df']
    update_P_E_simulation = params['update_p_e_simulation']
    update_WaterSip_DF_output = params['update_watersip_df_output']

#---------------------------------------------------------------------------   
    
    if update_DF:
        dynamic_rh_thresholds_calculation(method)
        dynamic_blh_factors_calculation(method) 

    start_time = time.time()
    filtering_and_tracking(method, 'on')  
    print (f'execution time: {time.time()-start_time:.4f} s')
    
    if update_WaterSip_DF_output:
        calculate_moisture_contribution('on')
    
    if update_P_E_simulation:
        p_simulation(method, 'on')
        e_simulation(method, 'on')
    
    bias_correct_moisture_contribution(method, 'on')
    
    print("WaterSip-DF-HAMSTER completed!")


if __name__ == "__main__":
    main_watersip_df_hamster() 