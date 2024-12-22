This is the Python-based code package that can be used to implement the WaterSip, WaterSip-HAMSTER, WaterSip-DF, and WaterSip-DF-HAMSTER methods. For detailed information on these methods, please refer to our GMD manuscript titled "Dynamic Particle Filtering in Lagrangian Model-Based Moisture Source-Receptor Diagnosis" by Li et al. 

This package consists of seven core scripts (Step 1–7), and their logic and inclusion relationships with the respective methods are illustrated in the following Figure, using dashed boxes in different colors (different functions require the participation of different core scripts). For example, the default WaterSip method only involves STEP 3 and 4, whereas the WaterSip-DF-HAMSTER approach requires all seven scripts (STEP 1–7). This code package is easy to use as a command-line tool, with all relevant parameter configurations editable in a standalone text document. In addition to using simple rectangular boundaries, this code package supports the direct use of "shapefile" (shp) files as the definition of target region. For the default WaterSip method, it allows the configuration of different Δq and RH thresholds, as well as BLH scaling factors. For the DF and bias-correction methods, precipitation and evaporation observation data can be replaced in a fixed format. This package also supports real-time output of filtered precipitation and evaporation particles, particle trajectory diagnostics, and global precipitation and evaporation simulations. For the final moisture source-receptor diagnostic outputs, it allows users to configure the spatial resolution.
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

#### Required data
1. Place the "partposit*" files (FLEXPART output under domain-filling mode) into the **partposit_file** folder.
2. Place the monthly "actual" precipitation/evaporation files (.nc format, e.g. ERA5 hourly data) into the **P_E_observation** folder.
3. Place the shapefile (shp) of the target area into the **shp** folder.

#### Configuration parameters
Please refer to and modify the **config.yaml** file.

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









