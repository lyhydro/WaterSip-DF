# -*- coding: utf-8 -*-

import os
import re
import numpy as np
import xarray as xr
import pandas as pd
import glob
import pickle
import re
from datetime import datetime, timedelta
import geopandas as gpd
from shapely.geometry import Point
import config


# %% * * * * * * * * * * * * FLEXPART output to E/P simulation functions * * * * * * * * * * * *
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


def get_and_combine_obs_files(observation_path, time_index, variable):
    result = []
    for t in time_index:
        pattern = os.path.join(observation_path, f'*{t}*.nc')
        ds = xr.open_dataset(glob.glob(pattern)[0])[variable]
        if 'valid_time' in ds.dims or 'valid_time' in ds.coords:
            ds = ds.rename({'valid_time': 'time'})
        result.append(ds)
    return xr.concat(result, dim='time')


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


def readpartposit_to_df(file, variables=None):
    
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
    
    with open(file, 'rb') as f:
        nbytes = os.fstat(f.fileno()).st_size
        numpart = round((nbytes / 4 + 3) / 15 - 1)
        f.read(4 * 3)  # skip the head
        data_as_int = np.fromfile(f, dtype=np.int32, count=numpart * 15).reshape(-1, 15).T
        f.seek(4 * 3)  # reload position
        data_as_float = np.fromfile(f, dtype=np.float32, count=numpart * 15).reshape(-1, 15).T
    
    if variables is None:
        variables = list(variable_map.keys())
    
    output_data = {}
    for var in variables:
        if var in variable_map:
            output_data[var] = data_as_float[variable_map[var], :]
    output = pd.DataFrame(output_data, index=data_as_int[1, :]) 
    return output


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


def write_to_nc_3d(data, var_name, file_name, time, latitude, longitude, output_path):
    result = xr.DataArray(data,
                          coords={"time": time, "latitude": latitude, "longitude": longitude},
                          dims=["time", "latitude", "longitude"],
                          name=var_name)
    result.to_netcdf(os.path.join(output_path, "{}.nc".format(file_name)))
    print(file_name, 'write to nc done.')


def write_to_nc_2d(data, var_name, file_name, latitude, longitude, output_path):
    result = xr.DataArray(data,
                          coords={"latitude": latitude, "longitude": longitude},
                          dims=["latitude", "longitude"],
                          name=var_name)
    result.to_netcdf(os.path.join(output_path, "{}.nc".format(file_name)))
    print(file_name, 'write to nc done.')


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


def calculate_RH(df, RH_name='RH', dens='dens', q='q', t='t'):

    p = df[dens] * 2.8705 * (1 + 0.608 * df[q]) * df[t]  # hpa, P=ρ⋅Rd(1+0.608q)⋅T
    df[RH_name] = 26.3 * p * df[q] / np.exp((17.67 * (df[t] - 273.16)) / (df[t] - 29.65))


def midpoint(lat1, lon1, lat2, lon2):

    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    x1, y1, z1 = np.cos(lat1) * np.cos(lon1), np.cos(lat1) * np.sin(lon1), np.sin(lat1)
    x2, y2, z2 = np.cos(lat2) * np.cos(lon2), np.cos(lat2) * np.sin(lon2), np.sin(lat2)
    
    x, y, z = (x1 + x2) / 2, (y1 + y2) / 2, (z1 + z2) / 2
    
    mid_lon = np.arctan2(y, x)
    mid_lat = np.arctan2(z, np.sqrt(x**2 + y**2))
    
    return np.degrees(mid_lat), np.degrees(mid_lon)


def point_distance(lat1, lon1, lat2, lon2):

    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371
    return c * r


def from_shp_get_parts(df, shp_file):

    gdf = gpd.read_file(shp_file)
    geometry = gdf.unary_union
    points = [Point(xy) for xy in zip(df['lon'], df['lat'])]
    mask = [geometry.contains(point) for point in points]
    return df[mask]


def from_bounds_get_parts(df, lat_up, lat_down, lon_left, lon_right):

    return df[(df['lat'] >= lat_down) & (df['lat'] <= lat_up) & 
              (df['lon'] >= lon_left) & (df['lon'] <= lon_right)]


def back_tracking_files(flexpart_output_file, start_time, tracking_days, time_span, key_string='partposit_'):

    start_date = datetime.strptime(start_time, '%Y%m%d%H')
    end_date = start_date - timedelta(days=tracking_days)
    files = []
    while start_date >= end_date:
        file_str = start_date.strftime('%Y%m%d%H0000')
        file_name = f'{key_string}{file_str}'
        file_path = os.path.join(flexpart_output_file, file_name)
        files.append(file_path)
        start_date = start_date - timedelta(hours=time_span)
    return files


def readpkl(file_path):

    with open(file_path, 'rb') as f:
        return pickle.load(f)


def writepkl(file_path, result):

    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'wb') as f:
        pickle.dump(result, f)


def from_shp_get_mask(shp, latitude, longitude):

    gdf = gpd.read_file(shp)
    geometry = gdf.unary_union
    
    mask = np.zeros((len(latitude), len(longitude)), dtype=bool)
    for i, lat in enumerate(latitude):
        for j, lon in enumerate(longitude):
            point = Point(lon, lat)
            mask[i, j] = geometry.contains(point)
    return mask


def from_bounds_get_mask(lat_up, lat_down, lon_left, lon_right, latitude, longitude):

    lat_mask = (latitude >= lat_down) & (latitude <= lat_up)
    lon_mask = (longitude >= lon_left) & (longitude <= lon_right)
    return np.outer(lat_mask, lon_mask)


def get_df_BLH_factor(DF_file_path, start_time, tracking_days, direction='backward'):

    pattern = re.compile(r"DF_BLH_factors_(\d{10})\.nc")
    end_time = (datetime.strptime(start_time, '%Y%m%d%H') - timedelta(days=tracking_days)).strftime('%Y%m%d%H')
    lists = os.listdir(DF_file_path)
    lists = sorted(lists)
    selected_files = [file for file in lists
                      if (match := pattern.search(file)) and end_time <= match.group(1) <= start_time]
    if selected_files[0][-13:-3] > end_time:
        print('! Lacking enough DF_BLH_factors files')
    df = []
    for f in selected_files:
        DF_BLH_factors = xr.open_dataset('{}\{}'.format(DF_file_path, f))['DF_BLH_factors']
        df0 = DF_BLH_factors.to_dataframe(name='BLH_factor').reset_index()
        df0 = df0.dropna(subset=['BLH_factor'])
        df0 = df0.query("BLH_factor != 0")
        df0 = df0.assign(time=pd.to_datetime(f[-13:-3], format='%Y%m%d%H'))
        df.append(df0)
    return pd.concat(df, ignore_index=True)


def check_file(file_path):

    return os.path.exists(file_path)


def calculate_coordinate_round(coordinate_value, output_spatial_resolution):

    return np.round(coordinate_value / output_spatial_resolution) * output_spatial_resolution


def get_algorithm_params(method):

    if method == 'WaterSip':
        return {
            'q_diff_p': config.WATERSIP_Q_DIFF_P,
            'q_diff_e': config.WATERSIP_Q_DIFF_E,
            'rh_threshold': config.WATERSIP_RH_THRESHOLD,
            'blh_factor': config.WATERSIP_BLH_FACTOR,
            'update_temporary': config.WATERSIP_UPDATE_TEMPORARY
        }
    elif method == 'WaterSip-DF':
        return {
            'q_diff_p': config.WATERSIP_DF_Q_DIFF_P,
            'q_diff_e': config.WATERSIP_DF_Q_DIFF_E,
            'update_df': config.WATERSIP_DF_UPDATE_DF,
            'update_temporary': config.WATERSIP_DF_UPDATE_TEMPORARY
        }
    elif method == 'WaterSip-HAMSTER':
        return {
            'q_diff_p': config.WATERSIP_HAMSTER_Q_DIFF_P,
            'q_diff_e': config.WATERSIP_HAMSTER_Q_DIFF_E,
            'rh_threshold': config.WATERSIP_HAMSTER_RH_THRESHOLD,
            'blh_factor': config.WATERSIP_HAMSTER_BLH_FACTOR,
            'update_p_e_simulation': config.WATERSIP_HAMSTER_UPDATE_P_E_SIMULATION,
            'update_temporary': config.WATERSIP_HAMSTER_UPDATE_TEMPORARY            
        }
    elif method == 'WaterSip-DF-HAMSTER':
        return {
            'q_diff_p': config.WATERSIP_DF_HAMSTER_Q_DIFF_P,
            'q_diff_e': config.WATERSIP_DF_HAMSTER_Q_DIFF_E,
            'update_df': config.WATERSIP_DF_HAMSTER_UPDATE_DF,
            'update_p_e_simulation': config.WATERSIP_DF_HAMSTER_UPDATE_P_E_SIMULATION,      
            'update_temporary': config.WATERSIP_DF_HAMSTER_UPDATE_TEMPORARY
        }
    else:
        raise ValueError(f"Unknown method: {method}") 