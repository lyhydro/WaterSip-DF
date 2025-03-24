This is the Python project that can be used to implement the WaterSip (Sodemann et al., 2008), WaterSip-HAMSTER (Keune et al., 2022), WaterSip-DF, and WaterSip-DF-HAMSTER methods. For detailed information on the DF methods, please refer to our manuscript titled "Dynamic Particle Filtering in Lagrangian Model-Based Moisture Source-Receptor Diagnosis" by Li et al. 

This project consists of seven core scripts (Step 1–7), and their logic and inclusion relationships with the respective methods are illustrated in the following Figure, using dashed boxes in different colors (different functions require the participation of different core scripts). For example, the default WaterSip method only involves STEP 3 and 4, whereas the WaterSip-DF-HAMSTER approach requires all seven scripts (STEP 1–7). This project is easy to use as a command-line tool, with relevant parameter configurations editable in the YAML file. 
![image](https://github.com/user-attachments/assets/25c0f30e-9201-4147-9af7-fed30a1d760c)

#### Install
To install this code package, do the following:
1. Clone the repository
```shell
git clone https://github.com/lyhydro/WaterSip-DF
cd WaterSip-DF
```
2. Make an environment with the necessary python packages
```shell
pip install -r requirements.txt
```

#### Configuration
Refer to and modify the **config.yaml** file.

#### Required data
1. Place the "partposit*" files (FLEXPART output under domain-filling mode) into the **partposit_file** folder.
2. If use WaterSip-HAMSTER, WaterSip-DF, or WaterSip-DF-HAMSTER, place precipitation/evaporation observation data (.nc format, e.g. ERA5 hourly data) into the **P_E_observation** folder.
3. If use shapefile to difine the target region, place .shp file into the **shp** folder.

#### Run the methods
```shell
# WaterSip
python main.py watersip
# WaterSip-DF
python main.py watersip-df
# WaterSip-HAMSTER
python main.py watersip-hamster
# WaterSip-DF-HAMSTER
python main.py watersip-df-hamster
```









