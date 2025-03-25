# -*- coding: utf-8 -*-
import sys
from WaterSip import main_watersip
from WaterSip_DF import main_watersip_df
from WaterSip_HAMSTER import main_watersip_hamster
from WaterSip_DF_HAMSTER import main_watersip_df_hamster

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == 'watersip':
        main_watersip()
    elif len(sys.argv) > 1 and sys.argv[1] == 'watersip-hamster':
        main_watersip_hamster()
    elif len(sys.argv) > 1 and sys.argv[1] == 'watersip-df':
        main_watersip_df()
    elif len(sys.argv) > 1 and sys.argv[1] == 'watersip-df-hamster':
        main_watersip_df_hamster()
    else:
        print('Check your command')
