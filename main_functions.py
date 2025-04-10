# -*- coding: utf-8 -*-
import os
import re
import numpy as np
import xarray as xr
import pandas as pd
import glob
import pickle
import argparse
from datetime import datetime, timedelta
import geopandas as gpd
from shapely.geometry import Point, shape
from YAMLConfig import YAMLConfig


# %% * * * * * * * * * * * * FLEXPART output to E/P simulation functions * * * * * * * * * * * *
# %%
def get_files(flexpart_output_file, start, end, time_span, key_string='partposit_'):
    start_date = datetime.strptime(start, '%Y%m%d%H')
    files = []
    while start_date <= datetime.strptime(end, '%Y%m%d%H'):
        file_str = start_date.strftime('%Y%m%d%H0000')
        file_name = f'{key_string}{file_str}'
        file_path = os.path.join(flexpart_output_file, file_name)
        files.append(file_path)
        start_date = start_date + timedelta(hours=time_span)
    return files

def get_files_temporary(flexpart_output_file, start, end, time_span, key_string='-6_P_particle_tracking_table.pkl'):
    start_date = datetime.strptime(start, '%Y%m%d%H')
    files = []
    while start_date <= datetime.strptime(end, '%Y%m%d%H'):
        file_str = start_date.strftime('%Y%m%d%H')
        file_name = f'{file_str}{key_string}'
        file_path = os.path.join(flexpart_output_file, file_name)
        files.append(file_path)
        start_date = start_date + timedelta(hours=time_span)
    return files


def get_files_new(flexpart_output_file: str, start: str, end: str, time_span: int,
                  key_string: str = 'partposit_') -> list:
    lists = os.listdir(flexpart_output_file)
    lists = sorted(lists)
    start_date = datetime.strptime(start, '%Y%m%d%H')
    end_date = datetime.strptime(end, '%Y%m%d%H')
    #
    date_list = [
        (start_date + timedelta(hours=hours)).strftime('%Y%m%d%H')
        for hours in
        range(0, (end_date - start_date).days * 24 + (end_date - start_date).seconds // 3600 + 1, time_span)
    ]
    #
    files = [
        os.path.join(flexpart_output_file, i)
        for i in lists
        if key_string in i and any(t in i for t in date_list)
    ]
    print(files)
    return files


# %%
def get_and_combine_obs_files(observation_path, time_index, variable):
    result = []
    for t in time_index:
        pattern = os.path.join(observation_path, f'*{t}*.nc')
        ds = xr.open_dataset(glob.glob(pattern)[0])[variable]
        if 'valid_time' in ds.dims or 'valid_time' in ds.coords:
            ds = ds.rename({'valid_time': 'time'})
        result.append(ds)
    return xr.concat(result, dim='time')


# %%
def readpartposit(file, nspec=1):
    with open(file, 'rb') as f:
        nbytes = os.fstat(f.fileno()).st_size
        numpart = round((nbytes / 4 + 3) / (14 + nspec) - 1)
        f.read(4 * 3)
        data_as_int = np.fromfile(f, dtype=np.int32, count=numpart * (14 + nspec)).reshape(-1, (14 + nspec)).T
        f.seek(4 * 3)
        data_as_float = np.fromfile(f, dtype=np.float32, count=numpart * (14 + nspec)).reshape(-1, (14 + nspec)).T
    output = {}
    output['npoint'] = data_as_int[1, :]  # point ID-number
    output['xlon'] = data_as_float[2, :]  # position coordinates
    output['ylat'] = data_as_float[3, :]  # position coordinates
    output['ztra'] = data_as_float[4, :]  # position coordinates
    output['itramem'] = data_as_int[5, :]  # relase times of each particle
    output['topo'] = data_as_float[6, :]  # Topography heigth in m
    output['pvi'] = data_as_float[7, :]  # Potential vorticity
    output['qvi'] = data_as_float[8, :]  # Specific humidity, kg/kg
    output['rhoi'] = data_as_float[9, :]  # Air density, kg/m*3
    output['hmixi'] = data_as_float[10, :]  # 'BLH' heigth in m (above ground surface)
    output['tri'] = data_as_float[11, :]  # Tropopause heigth in m
    output['tti'] = data_as_float[12, :]  # Temperature in K
    output['xmass'] = np.squeeze(data_as_float[13:13 + nspec, :])  # particle masses
    return output


# %%
def readpartposit_to_df(file, variables=None):
    #
    variable_map = {
        'lon': 2,
        'lat': 3,
        'z': 4,
        'q': 8,
        'dens': 9,
        'blh': 10,
        't': 12,
        'mass': 13
    }
    #
    with open(file, 'rb') as f:
        nbytes = os.fstat(f.fileno()).st_size
        numpart = round((nbytes / 4 + 3) / 15 - 1)
        f.read(4 * 3)  # skip the head
        data_as_int = np.fromfile(f, dtype=np.int32, count=numpart * 15).reshape(-1, 15).T
        f.seek(4 * 3)  # reload position
        data_as_float = np.fromfile(f, dtype=np.float32, count=numpart * 15).reshape(-1, 15).T
    # 
    if variables is None:
        variables = list(variable_map.keys())
    # 
    output_data = {}
    for var in variables:
        if var in variable_map:
            output_data[var] = data_as_float[variable_map[var], :]
    output = pd.DataFrame(output_data, index=data_as_int[1, :]) 
    return output


# %%
def global_gridcell_info(spatial_resolution, lat_nor=90, lat_sou=-90, lon_lef=-180, lon_rig=180):
    Erad = 6.371e6  # [m] Earth radius
    latitude = np.arange(lat_nor, lat_sou - spatial_resolution, -spatial_resolution)
    longitude = np.arange(lon_lef, lon_rig + spatial_resolution, spatial_resolution)
    lat_n_bound = np.minimum(lat_nor, latitude + 0.5 * spatial_resolution)
    lat_s_bound = np.maximum(lat_sou, latitude - 0.5 * spatial_resolution)
    gridcell_area = np.zeros([len(latitude), 1])
    gridcell_area[:, 0] = (np.pi / 180.0) * Erad ** 2 * abs(
        np.sin(lat_s_bound * np.pi / 180.0) - np.sin(lat_n_bound * np.pi / 180.0)) * spatial_resolution  # m^2
    gridcell_area = np.tile(gridcell_area, len(longitude))
    return latitude, longitude, gridcell_area


# %%
def write_to_nc_3d(data, var_name, file_name, time, latitude, longitude, output_path):
    result = xr.DataArray(data,
                          coords={"time": time, "latitude": latitude, "longitude": longitude},
                          dims=["time", "latitude", "longitude"],
                          name=var_name)
    result.to_netcdf(os.path.join(output_path, "{}.nc".format(file_name)))
    print(file_name, 'write to nc done.')


# %%
def write_to_nc_2d(data, var_name, file_name, latitude, longitude, output_path):
    result = xr.DataArray(data,
                          coords={"latitude": latitude, "longitude": longitude},
                          dims=["latitude", "longitude"],
                          name=var_name)
    result.to_netcdf(os.path.join(output_path, "{}.nc".format(file_name)))
    print(file_name, 'write to nc done.')


# %%
def write_to_nc(data, name, time, latitude, longitude, output_path):
    if isinstance(time, pd.DatetimeIndex):
        result = xr.DataArray(data,
                              coords={"time": time, "latitude": latitude, "longitude": longitude},
                              dims=["time", "latitude", "longitude"],
                              name=name)
        result.to_netcdf(os.path.join(output_path, "{}.nc".format(name)))
    else:
        result = xr.DataArray(data,
                              coords={"latitude": latitude, "longitude": longitude},
                              dims=["latitude", "longitude"],
                              name=name)
        result.to_netcdf(os.path.join(output_path, "{}.nc".format(name)))
    print(name, 'write to nc done.')


# %%
def calculate_RH(df, RH_name='RH', dens='dens', q='q', t='t'):
    p = df[dens] * 2.8705 * (1 + 0.608 * df[q]) * df[t]  # hpa, P=ρ⋅Rd(1+0.608q)⋅T
    # es = 6.112*np.exp((17.67*(df['t']-273.16))/(df['t']-29.65))
    # qs = 0.622*es/(p-0.378*es)
    # df['RH_new'] = 100*df['q']/qs
    df[RH_name] = 26.3 * p * df[q] / np.exp((17.67 * (df[t] - 273.16)) / (df[t] - 29.65))  # https://earthscience.stackexchange.com/questions/2360/how-do-i-convert-specific-humidity-to-relative-humidity


# %%
def midpoint(lat1, lon1, lat2, lon2):
    # 
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    # 
    x1, y1, z1 = np.cos(lat1) * np.cos(lon1), np.cos(lat1) * np.sin(lon1), np.sin(lat1)
    x2, y2, z2 = np.cos(lat2) * np.cos(lon2), np.cos(lat2) * np.sin(lon2), np.sin(lat2)
    # 
    x_mid, y_mid, z_mid = (x1 + x2) / 2, (y1 + y2) / 2, (z1 + z2) / 2
    # 
    hyp = np.sqrt(x_mid ** 2 + y_mid ** 2)
    mid_lat = np.degrees(np.arctan2(z_mid, hyp))
    mid_lon = np.degrees(np.arctan2(y_mid, x_mid))
    return mid_lat, mid_lon


# %%
def point_distance(lat1, lon1, lat2, lon2):
    # 
    R = 6371393
    # 
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)
    # 
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    # 
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    # 
    distance = R * c
    return distance


# %% * * * * * * * * * * * * tracking functions * * * * * * * * * * * *
# %%
def from_shp_get_parts(df, shp_file):
    try:
        gdf_shp = gpd.read_file(shp_file)
    except Exception as e:
        print(f"An error occurred: {e}")
    # 
    shp_bounds = gdf_shp.total_bounds
    df = df[(df['lon'] >= shp_bounds[0]) & (df['lon'] <= shp_bounds[2]) & (df['lat'] >= shp_bounds[1]) & (
            df['lat'] <= shp_bounds[3])]
    points = [Point(lon, lat) for lon, lat in zip(df['lon'], df['lat'])]
    gdf_points = gpd.GeoDataFrame(geometry=points, index=df.index, crs=gdf_shp.crs)
    points_within_shp = gpd.sjoin(gdf_points, gdf_shp, predicate='within')
    return df.loc[
        points_within_shp.index]  


# %%
def from_bounds_get_parts(df, lat_up, lat_down, lon_left, lon_right):
    df_filtered = df[(df['lat'] >= lat_down) & (df['lat'] <= lat_up) &
                     (df['lon'] >= lon_left) & (df['lon'] <= lon_right)]
    return df_filtered


# %%
def back_tracking_files(flexpart_output_file, start_time, tracking_days, time_span, key_string='partposit_'):
    start_date = datetime.strptime(start_time, '%Y%m%d%H')
    files = []
    for i in range(1 + tracking_days * 24 // time_span):
        current_time = start_date - timedelta(hours=i * time_span)
        file_str = current_time.strftime('%Y%m%d%H0000')
        file_name = f'{key_string}{file_str}'
        file_path = os.path.join(flexpart_output_file, file_name)
        files.append(file_path)
    return files


# %%
def readpkl(file_path):
    with open(file_path, 'rb') as content:
        return pickle.load(content)


# %%
def writepkl(file_path, result):
    with open(file_path, 'wb') as content:
        return pickle.dump(result, content)

    # %% * * * * * * * * * * * * other functions * * * * * * * * * * * *


# %%
def from_shp_get_mask(shp, latitude, longitude):
    gdf_shp = gpd.read_file(shp)
    mask = np.zeros((len(latitude), len(longitude)))
    shp_geom = shape(gdf_shp.geometry.unary_union)
    for i, lat in enumerate(latitude):
        for j, lon in enumerate(longitude):
            point = Point(lon, lat) 
            if point.within(shp_geom):  
                mask[i, j] = 1  
    return mask


# %%
def from_bounds_get_mask(lat_up, lat_down, lon_left, lon_right, latitude, longitude):
    lon_grid, lat_grid = np.meshgrid(longitude, latitude)
    mask = ((lat_grid >= lat_down) & (lat_grid <= lat_up) &
            (lon_grid >= lon_left) & (lon_grid <= lon_right)).astype(int)
    return mask


# %%
def get_df_BLH_factor(DF_file_path, start_time, tracking_days, direction='backward'):
    pattern = re.compile(r"DF_BLH_factors_(\d{10})\.nc")
    end_time = (datetime.strptime(start_time, '%Y%m%d%H') - timedelta(days=tracking_days)).strftime('%Y%m%d%H')
    lists = os.listdir(DF_file_path)
    lists = sorted(lists)
    selected_files = [file for file in lists
                      if (match := pattern.search(file)) and end_time <= match.group(1) <= start_time]
    if selected_files[0][-13:-3] > end_time:
        print(' * * * Lacking enough DF_BLH_factors files! * * *')
    df = []
    for f in selected_files:
        DF_BLH_factors = xr.open_dataset('{}\{}'.format(DF_file_path, f))['DF_BLH_factors']
        df0 = DF_BLH_factors.to_dataframe(name='BLH_factor').reset_index()
        df0 = df0.dropna(subset=['BLH_factor'])
        df0 = df0.query("BLH_factor != 0")
        df0 = df0.assign(time=pd.to_datetime(f[-13:-3], format='%Y%m%d%H'))
        df.append(df0)
    return pd.concat(df, ignore_index=True)

#%%
def check_file(file_path):
    return os.path.exists(file_path)


def gen_df_rh_or_p_simulation_files():
    config = YAMLConfig('config.yaml')
    general_config = config.get('General')
    time_span = general_config['time_span']
    start_time = str(general_config['start_time'])
    end_time = str(general_config['end_time'])

    DF_file_path = config.get('WaterSip-DF')['DF_file_path']

    time = pd.date_range(start=pd.to_datetime(start_time, format='%Y%m%d%H'),
                         end=pd.to_datetime(end_time, format='%Y%m%d%H'), freq='{}H'.format(time_span))[:-1]
    files = [os.path.join(DF_file_path, f"DF_RH_thresholds_{time.strftime('%Y%m%d%H')}.nc") for time in time]
    return files


#
def calculate_coordinate_round(coordinate_value, output_spatial_resolution):
    return (coordinate_value / output_spatial_resolution).round() * output_spatial_resolution


def read_cmdargs():
    """
    ACTION: read dates, thresholds and flags from command line
    RETURN: 'args' contains all
    DEP:    uses argparse
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--start_time",
        "-sti",
        help="start time",
        metavar="",
        type=str,
        default="2023073100",
    )
    parser.add_argument(
        "--end_time",
        "-eti",
        help="end time",
        metavar="",
        type=str,
        default="2023080100",
    )
    parser.add_argument(
        "--steps",
        "-st",
        help="steps performed (0: flex2traj, 1: diagnosis, 2: attribution, 3: bias correction)",
        metavar="",
        type=int,
        default=1,
    )
    args = parser.parse_args()  # namespace
    return args


if __name__ == "__main__":
    # result = check_p_e_simulation()
    # print(result)
    get_files('.\partposit_file', '2023070100', '2023080100', 9)
    get_files_new('.\partposit_file', '2023070100', '2023080100', 9)
