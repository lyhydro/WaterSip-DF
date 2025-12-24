# -*- coding: utf-8 -*-


# ==================== PATH ====================
PARTPOSIT_PATH = r'..\output-2018' # FLEXPART output (partposit*)
TEMPORARY_FILE_PATH = r'..\test' # Store temporary files (.pkl)
FINAL_OUTPUT_PATH = r'..\test' # Store final output (.nc)
# useful for DF method (WaterSip-DF & WaterSip-DF-HAMSTER)
DF_FILE_PATH = r'..\test'
# useful for HAMSTER method (WaterSip-HAMSTER & WaterSip-DF-HAMSTER)
OBSERVATION_PATH = r'..\P_E_observation' # currently only support ERA5
P_E_SIMULATION_OUTPUT_PATH = r'..\test'

# ==================== Basic Configration ====================
START_TIME = '2018060100' # 'YearMonthDayHour'
END_TIME = '2018090100' # 'YearMonthDayHour', not include time_span after the time
TRACKING_DAYS = 20 # days
TIME_SPAN = 6 # Time span of FLEXPART output (partposit*)
OUTPUT_SPATIAL_RESOLUTION = 1 # Degree of output grids
TARGET_REGION = r'..\TibetanPlateau.shp'  # r'.\shp\boundary.shp' or [lat_up, lat_down, lon_left, lon_right]

# ==================== WaterSip Configration ====================
WATERSIP_Q_DIFF_P = -0.0000  # kg/kg
WATERSIP_Q_DIFF_E = 0.0002 # kg/kg
WATERSIP_RH_THRESHOLD = 80 # %
WATERSIP_BLH_FACTOR = 1.5 # Scaling factor to BLH

# ==================== WaterSip-DF Configration ====================
WATERSIP_DF_Q_DIFF_P = -0.0000 # kg/kg
WATERSIP_DF_Q_DIFF_E = 0.0000 # kg/kg
WATERSIP_DF_UPDATE_DF = False # True or False

# ==================== WaterSip-HAMSTER Configration ====================
WATERSIP_HAMSTER_Q_DIFF_P = -0.0000 # kg/kg
WATERSIP_HAMSTER_Q_DIFF_E = 0.0002 # kg/kg
WATERSIP_HAMSTER_RH_THRESHOLD = 80 # %
WATERSIP_HAMSTER_BLH_FACTOR = 1.5 # Scaling factor to BLH
WATERSIP_HAMSTER_UPDATE_WATERSIP_OUTPUT = False # True or False, update WaterSip output (.nc)
WATERSIP_HAMSTER_UPDATE_P_E_SIMULATION = False # True or False, the P_E_SIMULATION need from WaterSip

# ==================== WaterSip-DF-HAMSTER Configration ====================
WATERSIP_DF_HAMSTER_Q_DIFF_P = -0.0000 # kg/kg
WATERSIP_DF_HAMSTER_Q_DIFF_E = 0.0000 # kg/kg
WATERSIP_DF_HAMSTER_UPDATE_WATERSIP_DF_OUTPUT = True # True or False, update WaterSip-DF output (.nc)
WATERSIP_DF_HAMSTER_UPDATE_P_E_SIMULATION = True # True or False, the P_E_SIMULATION need from WaterSip-DF
WATERSIP_DF_HAMSTER_UPDATE_DF = True # True or False