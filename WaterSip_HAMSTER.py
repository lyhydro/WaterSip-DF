# -*- coding: utf-8 -*-
from Step_3_Filtering_and_tracking import filtering_and_tracking
from Step_4_Calculate_moisture_contribution import calculate_moisture_contribution
from Step_5_P_simulation import p_simulation
from Step_6_E_simulation import e_simulation
from Step_7_Bias_correct_moisture_contribution import bias_correct_moisture_contribution

def main_watersip_hamster():
    # step 3
    filtering_and_tracking('off')
    # step 4
    calculate_moisture_contribution('off')
    # step 5
    p_simulation('off')
    # step 6
    e_simulation('off')
    # step 7
    bias_correct_moisture_contribution('off')
