# -*- coding: utf-8 -*-
from Step_3_Filtering_and_tracking import FilteringAndTracking
from Step_4_Calculate_moisture_contribution import CalculateMoistureContribution


def main_watersip():
    # step 3
    ft = FilteringAndTracking('WaterSip')
    ft.filtering_and_tracking('off')
    # step 4
    cmc = CalculateMoistureContribution()
    cmc.calculate_moisture_contribution('off')

