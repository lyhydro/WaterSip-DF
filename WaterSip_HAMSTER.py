# -*- coding: utf-8 -*-
from Step_3_Filtering_and_tracking import FilteringAndTracking
from Step_4_Calculate_moisture_contribution import CalculateMoistureContribution
from Step_5_P_simulation import PSimulation
from Step_6_E_simulation import ESimulation
from Step_7_Bias_correct_moisture_contribution import BiasCorrectMoistureContribution
from YAMLConfig import YAMLConfig


def main_watersip_hamster():
    config = YAMLConfig('config.yaml')
    update_P_E_simulation = config.get('WaterSip-HAMSTER')['update_P_E_simulation']
    
    # step 3
    ft = FilteringAndTracking('WaterSip-HAMSTER')
    ft.filtering_and_tracking('off')
    # step 4
    cmc = CalculateMoistureContribution()
    cmc.calculate_moisture_contribution('off')
    
    if update_P_E_simulation:
        # step 5
        PSimulation('WaterSip-HAMSTER').p_simulation('off')
        # step 6
        ESimulation('WaterSip-HAMSTER').e_simulation('off')
    # step 7
    BiasCorrectMoistureContribution('WaterSip-HAMSTER').bias_correct_moisture_contribution('off')
