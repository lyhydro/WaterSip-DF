# -*- coding: utf-8 -*-
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib.colors import LinearSegmentedColormap

#%% * * * * * * * * * * * * * * * * * STEP 1 : DF_RH_thresholds * * * * * * * * * * * * * * * * * 
DF_RH_thresholds_path = r'D:\Articles\WaterSip-DF\code-new\temporary\DF_RH_thresholds.nc'
DF_RH_thresholds = xr.open_dataset( DF_RH_thresholds_path )['DF_RH_thresholds'] #.resample(time='6H').sum(dim='time')*1000
latitude = DF_RH_thresholds['latitude']
longitude = DF_RH_thresholds['longitude']
DF_RH_thresholds_mean = DF_RH_thresholds.mean(dim='time', skipna=True) 
#
fig = plt.figure(figsize=(10, 6), dpi=300)
m = Basemap(projection='cyl', llcrnrlat=-90, urcrnrlat=90, llcrnrlon=-179, urcrnrlon=180, resolution='c')
m.drawmapboundary(fill_color='white')
m.drawmeridians(np.arange(-150,151,30),linewidth=2,labels=[0,0,0,1],dashes=[1.5,999], fontsize = 11)
m.drawparallels(np.arange(-60,61,30),linewidth=2,labels=[1,0,0,0],dashes=[1.5,999],fontsize = 11)
shapeinfo = m.readshapefile(r'D:\Articles\1998 2020 moisture to precipitation MLYRB\ARCGIS\ne_10m_coastline','FID',linewidth=0.5, color='k')
shapeinfo = m.readshapefile("D:\\Articles\\Oceanic Evaporation to TP\\ARCGIS\\TibetanPlateau",'FID',linewidth=0.8, color='k')
mesh_lon, mesh_lat = np.meshgrid(longitude,  latitude)
ticks = np.linspace(30,90,5) #改范围啊
im = m.pcolor(mesh_lon, mesh_lat, DF_RH_thresholds_mean, cmap='GnBu')
im.set_clim(30,90) #改范围啊
bar = fig.colorbar(im, cax=fig.add_axes([0.125, 0.1, 0.7, 0.025]), orientation='horizontal', ticks=ticks, extend='max')
bar.ax.tick_params(labelsize=11)
fig.text(0.83, 0.07,  '%', fontsize=12)    
fig.savefig( r'D:\Articles\WaterSip-DF\WaterSip-DF-HAMSTER\output\DF_RH_thresholds.tif', bbox_inches='tight' )
plt.close()

#%% * * * * * * * * * * * * * * * * * STEP 2 : DF_BLH_factors * * * * * * * * * * * * * * * * * 
DF_BLH_factors_path = r'D:\Articles\WaterSip-DF\code-new\temporary\DF_BLH_factors.nc'
DF_BLH_factors = xr.open_dataset( DF_BLH_factors_path )['DF_BLH_factors'] #.resample(time='6H').sum(dim='time')*1000
latitude = DF_BLH_factors['latitude']
longitude = DF_BLH_factors['longitude']
DF_BLH_factors_mean = DF_BLH_factors.mean(dim='time', skipna=True) 
#
fig = plt.figure(figsize=(10, 6), dpi=300)
m = Basemap(projection='cyl', llcrnrlat=-90, urcrnrlat=90, llcrnrlon=-179, urcrnrlon=180, resolution='c')
m.drawmapboundary(fill_color='white')
m.drawmeridians(np.arange(-150,151,30),linewidth=2,labels=[0,0,0,1],dashes=[1.5,999], fontsize = 11)
m.drawparallels(np.arange(-60,61,30),linewidth=2,labels=[1,0,0,0],dashes=[1.5,999],fontsize = 11)
shapeinfo = m.readshapefile(r'D:\Articles\1998 2020 moisture to precipitation MLYRB\ARCGIS\ne_10m_coastline','FID',linewidth=0.5, color='k')
shapeinfo = m.readshapefile("D:\\Articles\\Oceanic Evaporation to TP\\ARCGIS\\TibetanPlateau",'FID',linewidth=0.8, color='k')
mesh_lon, mesh_lat = np.meshgrid(longitude,  latitude)
ticks = np.linspace(0.4,4.0,7) #改范围啊
im = m.pcolor(mesh_lon, mesh_lat, DF_BLH_factors_mean, cmap='GnBu')
im.set_clim(0.2,4.2) #改范围啊
bar = fig.colorbar(im, cax=fig.add_axes([0.125, 0.1, 0.7, 0.025]), orientation='horizontal', ticks=ticks, extend='both')
bar.ax.tick_params(labelsize=11)
fig.text(0.83, 0.07, 'factor', fontsize=12)  
fig.savefig( r'D:\Articles\WaterSip-DF\WaterSip-DF-HAMSTER\output\DF_BLH_factors.tif', bbox_inches='tight' )
plt.close()

#%% * * * * * * * * * * * * * * * * * STEP 4 & 7 : moisture_contribution_mm * * * * * * * * * * * * * * * * * 
moisture_contribution_output_path = r'D:\Data\WaterSip-DF-HAMSTER\output\moisture_contribution_mm_2023073100_2023080100.nc'
#
moisture_contribution_mm = xr.open_dataset( moisture_contribution_output_path )['moisture_contribution_mm']
if len(moisture_contribution_mm.dims) == 3:
    moisture_contribution_mm = moisture_contribution_mm.sum(dim='time')
latitude = moisture_contribution_mm['latitude']
longitude = moisture_contribution_mm['longitude']

cdict = {'red': ((0., 1, 1),(0.05, 1, 1),(0.11, 0, 0),(0.66, 1, 1),(0.89, 1, 1),(1, 0.5, 0.5),(1, 0, 0)),
         'green': ((0., 1, 1),(0.05, 1, 1),(0.11, 0, 0),(0.375, 1, 1),(0.64, 1, 1),(0.91, 0, 0),(1, 0, 0)),
         'blue': ((0., 1, 1),(0.05, 1, 1),(0.11, 1, 1),(0.34, 1, 1),(0.65, 0, 0),(1, 0, 0))}
my_cmap = LinearSegmentedColormap('my_colormap',cdict,256)
cmap = LinearSegmentedColormap.from_list('my_cmap', [(0 , 'lightyellow'),
                                                     (0.02, 'orange'),
                                                     (0.1, 'm'),
                                                     (0.3, 'purple'),
                                                     (1, 'indigo')])
fig = plt.figure(figsize=(10, 6), dpi=150)
m = Basemap(projection='cyl', llcrnrlat=-90, urcrnrlat=90, llcrnrlon=-179, urcrnrlon=180, resolution='c')
m.drawmapboundary(fill_color='white')
m.drawmeridians(np.arange(-150,151,30),linewidth=2,labels=[0,0,0,1],dashes=[1.5,999], fontsize = 11)
m.drawparallels(np.arange(-60,61,30),linewidth=2,labels=[1,0,0,0],dashes=[1.5,999],fontsize = 11)
shapeinfo = m.readshapefile(r'D:\Articles\1998 2020 moisture to precipitation MLYRB\ARCGIS\ne_10m_coastline','FID',linewidth=0.5, color='k')
shapeinfo = m.readshapefile("D:\\Articles\\Oceanic Evaporation to TP\\ARCGIS\\TibetanPlateau",'FID',linewidth=0.8, color='k')
shapeinfo = m.readshapefile("D:\\Articles\\WaterSip-DF\\YRB\shp\\boundary",'FID',linewidth=0.8, color='cyan')
mesh_lon, mesh_lat = np.meshgrid(longitude,  latitude)

clevs = [0.01,0.1,0.2,0.47,0.73,1,1.33,1.66,2,2.6,3.2,4,5.2,6.4,8,10,13,16,21,26,32,48,64]  
clevs2 = [0.2,1,2,4,8,16,32]
moisture_contribution_mm = moisture_contribution_mm.where(moisture_contribution_mm >= 0.005, other=np.nan)
im = m.contourf(mesh_lon, mesh_lat, moisture_contribution_mm,clevs,linewidths=1.5,latlon=True, cmap=cmap, extend='both')
bar = fig.colorbar(im, cax=fig.add_axes([0.125, 0.1, 0.7, 0.025]), orientation='horizontal', ticks=clevs2, extend='max')
bar.ax.tick_params(labelsize=11)
fig.text(0.83, 0.07,  'mm', fontsize=12) 
fig.savefig( r'{}.tif'.format(moisture_contribution_output_path[:-3]), bbox_inches='tight' )
plt.close()

