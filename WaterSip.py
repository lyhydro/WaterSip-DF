# -*- coding: utf-8 -*-

from Step_3_Filtering_and_tracking import filtering_and_tracking
from Step_4_Calculate_moisture_contribution import calculate_moisture_contribution


def main_watersip():

    print("Starting WaterSip ...")
    
    method = 'WaterSip'

#---------------------------------------------------------------------------   
    
    # Step 3: Filtering and tracking
    filtering_and_tracking(method, 'off')
    
    # Step 4: Calculate moisture contribution
    calculate_moisture_contribution('off')
    
    print("WaterSip completed!")


if __name__ == "__main__":
    main_watersip() 