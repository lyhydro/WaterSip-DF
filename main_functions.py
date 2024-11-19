# -*- coding: utf-8 -*-
import os
import numpy as np
import xarray as xr
import pandas as pd
import pickle
from datetime import datetime, timedelta
import geopandas as gpd
from shapely.geometry import Point, shape

#%% * * * * * * * * * * * * FLEXPART output to E/P simulation functions * * * * * * * * * * * *
#%%
def get_files(flexpart_output_file, start, end, time_span, key_string='partposit_'):
    lists = os.listdir(flexpart_output_file)   
    start_date = datetime.strptime(start, '%Y%m%d%H') #- timedelta(hours=time_span) #根据start设定，看是否多取了向前一步的时间，便于计算diff
    end_date = datetime.strptime(end, '%Y%m%d%H')
    date = []
    while start_date<=end_date:
        date.append(start_date.strftime('%Y%m%d%H'))
        start_date += timedelta(hours=time_span)   
    files = []
    for i in lists:
        for t in date:            
            if key_string in i and t in i:
                files.append(os.path.join(flexpart_output_file,i))
    return files

#%%
def readpartposit(file, nspec=1):
    with open(file, 'rb') as f:
        nbytes = os.fstat(f.fileno()).st_size
        numpart = round((nbytes/4+3)/(14+nspec)-1)
        f.read(4*3)
        data_as_int = np.fromfile(f, dtype=np.int32, count=numpart*(14 + nspec)).reshape(-1, (14 + nspec)).T
        f.seek(4*3) 
        data_as_float = np.fromfile(f, dtype=np.float32, count=numpart*(14 + nspec)).reshape(-1, (14 + nspec)).T
    output = {}
    output['npoint'] = data_as_int[1, :] # point ID-number
    output['xlon'] = data_as_float[2, :] # position coordinates
    output['ylat'] = data_as_float[3, :] # position coordinates
    output['ztra'] = data_as_float[4, :] # position coordinates
    output['itramem'] = data_as_int[5, :] # relase times of each particle
    output['topo'] = data_as_float[6, :] # Topography heigth in m
    output['pvi'] = data_as_float[7, :] # Potential vorticity
    output['qvi'] = data_as_float[8, :] # Specific humidity, kg/kg
    output['rhoi'] = data_as_float[9, :] # Air density, kg/m*3
    output['hmixi'] = data_as_float[10, :] # 'PBL' heigth in m (above ground surface)
    output['tri'] = data_as_float[11, :] # Tropopause heigth in m
    output['tti'] = data_as_float[12, :] # Temperature in K   
    output['xmass'] = np.squeeze( data_as_float[13:13+nspec, :] ) # particle masses
    return output

#%%
def global_gridcell_info(spatial_resolution, lat_nor=90, lat_sou=-90, lon_lef=-180, lon_rig=180):
    Erad = 6.371e6 # [m] Earth radius
    latitude = np.arange(lat_nor, lat_sou-spatial_resolution, -spatial_resolution)
    longitude = np.arange(lon_lef, lon_rig+spatial_resolution, spatial_resolution)    
    lat_n_bound = np.minimum(lat_nor , latitude + 0.5*spatial_resolution)
    lat_s_bound = np.maximum(lat_sou , latitude - 0.5*spatial_resolution)    
    gridcell_area = np.zeros([len(latitude),1]) 
    gridcell_area[:,0] = (np.pi/180.0)*Erad**2 * abs( np.sin(lat_s_bound*np.pi/180.0) - np.sin(lat_n_bound*np.pi/180.0) ) * spatial_resolution #m^2 
    gridcell_area = np.tile(gridcell_area,len(longitude))
    return latitude, longitude, gridcell_area

#%%
def write_to_nc(data, name, time, latitude, longitude, output_path):
    if isinstance(time, pd.DatetimeIndex):
        result = xr.DataArray(data,
            coords = {"time": time, "latitude": latitude,"longitude": longitude},
            dims = ["time", "latitude", "longitude"],
            name = name)
        result.to_netcdf( os.path.join(output_path, "{}.nc".format(name)) )
    else:        
        result = xr.DataArray(data,
            coords = {"latitude": latitude,"longitude": longitude},
            dims = ["latitude", "longitude"],
            name = name)
        result.to_netcdf( os.path.join(output_path, "{}.nc".format(name)) )
    print( name, 'write to nc done.')
        
#%%
def calculate_RH(df, RH_name='RH', dens='dens', q='q', t='t'):
    p = df[dens]*2.8705*(1+0.608*df[q])*df[t] #hpa, P=ρ⋅Rd(1+0.608q)⋅T
    # es = 6.112*np.exp((17.67*(df['t']-273.16))/(df['t']-29.65))
    # qs = 0.622*es/(p-0.378*es)
    # df['RH_new'] = 100*df['q']/qs
    df[RH_name] =  26.3*p*df[q] / np.exp((17.67*(df[t]-273.16))/(df[t]-29.65)) #https://earthscience.stackexchange.com/questions/2360/how-do-i-convert-specific-humidity-to-relative-humidity
    return df

#%%
def midpoint(lat1, lon1, lat2, lon2):
    # 转换为弧度
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    # 计算笛卡尔坐标
    x1, y1, z1 = np.cos(lat1) * np.cos(lon1), np.cos(lat1) * np.sin(lon1), np.sin(lat1)
    x2, y2, z2 = np.cos(lat2) * np.cos(lon2), np.cos(lat2) * np.sin(lon2), np.sin(lat2)
    # 计算中点的笛卡尔坐标
    x_mid, y_mid, z_mid = (x1 + x2) / 2, (y1 + y2) / 2, (z1 + z2) / 2
    # 计算中点的地理坐标
    hyp = np.sqrt(x_mid ** 2 + y_mid ** 2)
    mid_lat = np.degrees(np.arctan2(z_mid, hyp))
    mid_lon = np.degrees(np.arctan2(y_mid, x_mid))
    return mid_lat, mid_lon

#%%
def point_distance(lat1, lon1, lat2, lon2):
    # 地球半径（m）
    R = 6371393
    # 将输入的纬度和经度转换为弧度
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)
    # 计算纬度和经度差
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    # 哈弗萨因公式
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    # 计算距离
    distance = R * c
    return distance

#%% * * * * * * * * * * * * tracking functions * * * * * * * * * * * *
#%%
def from_shp_get_parts(df,shp_file):
    gdf_shp = gpd.read_file(shp_file)
    #先将df按矩形框筛选一遍，减少后期计算量
    shp_bounds = gdf_shp.total_bounds
    df = df[(df['lon']>=shp_bounds[0]) & (df['lon']<=shp_bounds[2]) & (df['lat']>=shp_bounds[1]) & (df['lat']<=shp_bounds[3])]
    points = [Point(lon, lat) for lon, lat in zip(df['lon'], df['lat'])]
    gdf_points = gpd.GeoDataFrame(geometry=points, index=df.index, crs=gdf_shp.crs)
    points_within_shp = gpd.sjoin(gdf_points, gdf_shp, predicate='within')
    return df.loc[points_within_shp.index] #.loc是基于标签索引，.iloc是基于位置索引，两者都可以基于索引号df.loc[df.index=='']，df.iloc[df.index.get_loc('')]

#%%
def back_rtacking_files(flexpart_output_file, start_time, tracking_days, time_span):
    lists = os.listdir(flexpart_output_file)
    find_file = 'partposit_'+start_time
    position = [index for index, item in enumerate(lists) if find_file in item]
    files = []
    for i in range(int(position[0]-tracking_days*24/time_span) ,position[0]+1):
        files.append(os.path.join(flexpart_output_file,lists[i]))
    return files

#%%
def readpkl(file_path):
    with open(file_path, 'rb') as content:
        return pickle.load(content)

#%%
def writepkl(file_path,result):
    with open(file_path, 'wb') as content:
        return pickle.dump(result, content)  

#%% * * * * * * * * * * * * other functions * * * * * * * * * * * *
#%%
def from_shp_get_mask(shp, latitude, longitude):
    gdf_shp = gpd.read_file(shp)
    mask = np.zeros((len(latitude), len(longitude)))
    shp_geom = shape(gdf_shp.geometry.unary_union) 
    for i, lat in enumerate(latitude):
        for j, lon in enumerate(longitude):
            point = Point(lon, lat)  # 构建点
            if point.within(shp_geom):  # 判断点是否在shp文件的范围内
                mask[i, j] = 1  # 在shp范围内的点赋值为1
    return mask

