# -*- coding: utf-8 -*-
from Step_1_Dynamic_RH_thresholds_calculation import DynamicRHThresholdsCalculation
from Step_2_Dynamic_BLH_factors_calculation import DynamicBLHThresholdsCalculation
from Step_3_Filtering_and_tracking import FilteringAndTracking
from Step_4_Calculate_moisture_contribution import CalculateMoistureContribution
from Step_5_P_simulation import PSimulation
from Step_6_E_simulation import ESimulation
from Step_7_Bias_correct_moisture_contribution import BiasCorrectMoistureContribution
from YAMLConfig import YAMLConfig


def main_watersip_df_hamster():
    config = YAMLConfig('config.yaml')
    update_DF = config.get('WaterSip-DF-HAMSTER')['update_DF']
    update_P_E_simulation = config.get('WaterSip-DF-HAMSTER')['update_P_E_simulation']
    #
    if update_DF:
        # step 1
        DynamicRHThresholdsCalculation('WaterSip-DF-HAMSTER').dynamic_rh_thresholds_calculation()
        # step 2
        DynamicBLHThresholdsCalculation('WaterSip-DF-HAMSTER').dynamic_blh_factors_calculation()
    # step 3
    ft = FilteringAndTracking('WaterSip-DF-HAMSTER')
    ft.filtering_and_tracking('on')
    # step 4
    cmc = CalculateMoistureContribution()
    cmc.calculate_moisture_contribution('on')
    
    # 
    if update_P_E_simulation:
        # step 5
        PSimulation('WaterSip-DF-HAMSTER').p_simulation('on')
        # step 6
        ESimulation('WaterSip-DF-HAMSTER').e_simulation('on')
    # step 7
    BiasCorrectMoistureContribution('WaterSip-DF-HAMSTER').bias_correct_moisture_contribution('on')
