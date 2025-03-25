# -*- coding: utf-8 -*-
from Step_1_Dynamic_RH_thresholds_calculation import DynamicRHThresholdsCalculation
from Step_2_Dynamic_BLH_factors_calculation import DynamicBLHThresholdsCalculation
from Step_3_Filtering_and_tracking import FilteringAndTracking
from Step_4_Calculate_moisture_contribution import CalculateMoistureContribution
from YAMLConfig import YAMLConfig


def main_watersip_df():
    config = YAMLConfig('config.yaml')
    update_DF = config.get('WaterSip-DF')['update_DF']
    # 
    if update_DF:
        # step 1
        DynamicRHThresholdsCalculation('WaterSip-DF').dynamic_rh_thresholds_calculation()
        # step 2
        DynamicBLHThresholdsCalculation('WaterSip-DF').dynamic_blh_factors_calculation()
    # step 3
    ft = FilteringAndTracking('WaterSip-DF')
    ft.filtering_and_tracking('on')
    # step 4
    cmc = CalculateMoistureContribution()
    cmc.calculate_moisture_contribution('on')


# if __name__ == "__main__":
#     main_watersip_df()
#     # filtering_and_tracking('on')
