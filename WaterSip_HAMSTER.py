# -*- coding: utf-8 -*-
from Step_3_Filtering_and_tracking import filtering_and_tracking
from Step_4_Calculate_moisture_contribution import calculate_moisture_contribution
from Step_5_P_simulation import p_simulation
from Step_6_E_simulation import e_simulation
from Step_7_Bias_correct_moisture_contribution import bias_correct_moisture_contribution
from main_functions import check_p_e_simulation
from YAMLConfig import YAMLConfig


def main_watersip_hamster():
    config = YAMLConfig('config.yaml')
    update_P_E_simulation = config.get('WaterSip-HAMSTER')['update_P_E_simulation']
    
    result = check_p_e_simulation()
    # step 3
    filtering_and_tracking('off')
    # step 4
    calculate_moisture_contribution('off')
    
    if not result or update_P_E_simulation:
        # step 5
        p_simulation('off')
        # step 6
        e_simulation('off')
    # step 7
    bias_correct_moisture_contribution('off')
