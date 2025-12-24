# -*- coding: utf-8 -*-

import sys

if __name__ == "__main__":
    usage_msg = 'USAGE: python main.py [watersip|watersip-df|watersip-hamster|watersip-df-hamster]'
    
    if len(sys.argv) > 1:
        algorithm = sys.argv[1].lower()
        
        if algorithm == 'watersip':
            from WaterSip import main_watersip
            main_watersip()
        elif algorithm == 'watersip-df':
            from WaterSip_DF import main_watersip_df
            main_watersip_df()
        elif algorithm == 'watersip-hamster':
            from WaterSip_HAMSTER import main_watersip_hamster
            main_watersip_hamster()
        elif algorithm == 'watersip-df-hamster':
            from WaterSip_DF_HAMSTER import main_watersip_df_hamster
            main_watersip_df_hamster()
        else:
            print(usage_msg)
    else:
        print(usage_msg) 