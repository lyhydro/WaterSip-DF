# -*- coding: utf-8 -*-

from Step_3_Filtering_and_tracking import filtering_and_tracking
from Step_4_Calculate_moisture_contribution import calculate_moisture_contribution
from Step_5_P_simulation import p_simulation
from Step_6_E_simulation import e_simulation
from Step_7_Bias_correct_moisture_contribution import bias_correct_moisture_contribution
from functions import get_algorithm_params


def main_watersip_hamster():

    print("Starting WaterSip-HAMSTER ...")
    
    method = 'WaterSip-HAMSTER'
    params = get_algorithm_params(method)
    
    update_P_E_simulation = params['update_p_e_simulation']
    update_WaterSip_output = params['update_watersip_output']

#---------------------------------------------------------------------------    

    # Step 3: Filtering and tracking
    filtering_and_tracking(method, 'off') 
    
    if update_WaterSip_output:
        # Step 4: Calculate moisture contribution
        calculate_moisture_contribution('off')
    
    if update_P_E_simulation:
        # Step 5: P simulation
        p_simulation(method, 'off')
        
        # Step 6: E simulation
        e_simulation(method, 'off')
    
    # Step 7: Bias correct moisture contribution
    bias_correct_moisture_contribution(method, 'off')
    
    print("WaterSip-HAMSTER completed!")


if __name__ == "__main__":
    main_watersip_hamster() 