import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.basemap import Basemap
import numpy as np
import string
import cmocean as cm
import xarray as xr
import warnings
warnings.filterwarnings("ignore")

R4 = xr.open_dataset("./Validation_preds_R4/Errors_Complete_Chronos_Daily.nc")
R1 = xr.open_dataset("./Validation_preds_R1/Errors_Complete_Chronos_Daily.nc")

# Define dimensions for the figure
#w = 0.23
#h = 0.22
w= 0.35
h=0.35
X_t, Y_t = np.meshgrid(R1.lon.values, R1.lat.values)
lonsH, latsH= np.meshgrid(R4.lon.values, R4.lat.values)
#list_o = [[0, 0.902, w, 0.22], [0.40, 0.902, w, 0.22], [0.80, 0.902, w, 0.22], [1.2, 0.902, w, 0.22], [1.6, 0.902, w, 0.22],[2.0, 0.902, w-0.3, 0.22],
#          [0, 0.652, w, 0.22], [0.40, 0.652, w, 0.22], [0.80, 0.652, w, 0.22], [1.2, 0.652, w, 0.22], [1.6, 0.652, w, 0.22],[2.0, 0.652, w-0.3, 0.22]]
list_o = [[0, 0.902, w, 0.22], [0.40, 0.902, w, 0.22], [0.80, 0.902, w, 0.22], [1.2, 0.902, w, 0.22],[1.6, 0.902, w, 0.22],[2.0, 0.786, w-0.3, 0.29],
          [0, 0.742, w, 0.22], [0.40, 0.742, w, 0.22], [0.80, 0.742, w, 0.22], [1.2, 0.742, w, 0.22],[1.6, 0.742, w, 0.22],#[2.0, 0.782, w-0.3, 0.14],
          ]

fig = plt.figure(figsize=(9, 24))  # size of the figure

vmin= 0.4;vmax=0.8

# Figure 1
ax = plt.axes(list_o[0])
ax.text(0., 1.01, '(' + string.ascii_uppercase[0] + ')', fontsize=20, fontweight='bold', transform=ax.transAxes)
ax.set_title("Day 1", fontsize=16, fontweight='bold')
m = Basemap(projection='cyl', llcrnrlon=-68, llcrnrlat=10.5, urcrnrlon=-66.1, urcrnrlat=12.4, resolution='h')

h = m.pcolormesh(X_t, Y_t, R1.isel(time=0).MAE.values, shading='gouraud', 
                 latlon=True, vmin=vmin, vmax=vmax, cmap='nipy_spectral')
m.fillcontinents(color='lightgray')
m.drawcoastlines(color='black', linewidth=1)
m.drawparallels(np.arange(5, 15, 0.5), labels=[1, 0, 0, 0], linewidth=0.01, fontsize=14)
m.drawmeridians(np.arange(-180., 180., 0.5), labels=[0, 0, 0, 1], linewidth=0.01, fontsize=14)
ax.text(-0.3, 0.5, 'Region 1', fontsize=16, fontweight='bold', rotation=90, transform=ax.transAxes, va='center', ha='center')


# Figure 2
ax = plt.axes(list_o[1])
ax.text(0., 1.01, '(' + string.ascii_uppercase[1] + ')', fontsize=20, fontweight='bold', transform=ax.transAxes)
ax.set_title("Day 7", fontsize=16, fontweight='bold')
h = m.pcolormesh(X_t, Y_t, R1.isel(time=6).MAE.values, shading='gouraud', 
                 latlon=True, vmin=vmin, vmax=vmax, cmap='nipy_spectral')
m.fillcontinents(color='lightgray')
m.drawcoastlines(color='black', linewidth=1)
m.drawmeridians(np.arange(-180., 180., 0.5), labels=[0, 0, 0, 1], linewidth=0.01, fontsize=14)


# Figure 3
ax = plt.axes(list_o[2])
ax.text(0., 1.01, '(' + string.ascii_uppercase[2] + ')', fontsize=20, fontweight='bold', transform=ax.transAxes)
ax.set_title("Day 14", fontsize=16, fontweight='bold')
h = m.pcolormesh(X_t, Y_t, R1.isel(time=13).MAE.values, shading='gouraud', 
                 latlon=True, vmin=vmin, vmax=vmax, cmap='nipy_spectral')
m.fillcontinents(color='lightgray')
m.drawcoastlines(color='black', linewidth=1)
m.drawmeridians(np.arange(-180., 180., 0.5), labels=[0, 0, 0, 1], linewidth=0.01, fontsize=14)

# Figure 4
ax = plt.axes(list_o[3])
ax.text(0., 1.01, '(' + string.ascii_uppercase[3] + ')', fontsize=20, fontweight='bold', transform=ax.transAxes)
ax.set_title("Day 21", fontsize=16, fontweight='bold')
h = m.pcolormesh(X_t, Y_t, R1.isel(time=20).MAE.values, shading='gouraud', 
                 latlon=True, vmin=vmin, vmax=vmax, cmap='nipy_spectral')
m.fillcontinents(color='lightgray')
m.drawcoastlines(color='black', linewidth=1)
m.drawmeridians(np.arange(-180., 180., 0.5), labels=[0, 0, 0, 1], linewidth=0.01, fontsize=14)

# Figure 5
ax = plt.axes(list_o[4])
ax.text(0., 1.01, '(' + string.ascii_uppercase[4] + ')', fontsize=20, fontweight='bold', transform=ax.transAxes)
ax.set_title("Day 28", fontsize=16, fontweight='bold')
h = m.pcolormesh(X_t, Y_t, R1.isel(time=27).MAE.values, shading='gouraud', 
                 latlon=True, vmin=vmin, vmax=vmax, cmap='nipy_spectral')
m.fillcontinents(color='lightgray')
m.drawcoastlines(color='black', linewidth=1)
m.drawmeridians(np.arange(-180., 180., 0.5), labels=[0, 0, 0, 1], linewidth=0.01, fontsize=14)

# Colorbar
ax_cbar = plt.axes(list_o[5])
cbar = plt.colorbar(h, cax=ax_cbar, label='SST (°C)', orientation='vertical')
cbar.set_label('MAE (°C)', fontsize=17)
cbar.ax.tick_params(labelsize=17)


# Figure 7
ax = plt.axes(list_o[6])
ax.text(0., 1.01, '(' + string.ascii_uppercase[6] + ')', fontsize=20, fontweight='bold', transform=ax.transAxes)
ax.set_title("Day 1", fontsize=16, fontweight='bold')
m = Basemap(projection='cyl', llcrnrlon=-73.24, llcrnrlat=11.76, urcrnrlon=-71.35, urcrnrlat=13.65, resolution='h')

h = m.pcolormesh(lonsH, latsH, R4.isel(time=0).MAE.values, shading='gouraud', 
                 latlon=True, vmin=vmin, vmax=vmax, cmap='nipy_spectral')
m.fillcontinents(color='lightgray')
m.drawcoastlines(color='black', linewidth=1)
m.drawparallels(np.arange(5, 15, 0.5), labels=[1, 0, 0, 0], linewidth=0.01, fontsize=14)
m.drawmeridians(np.arange(-180., 180., 0.5), labels=[0, 0, 0, 1], linewidth=0.01, fontsize=14)
ax.text(-0.3, 0.5, 'Region 4', fontsize=16, fontweight='bold', rotation=90, transform=ax.transAxes, va='center', ha='center')

# Figure 8
ax = plt.axes(list_o[7])
ax.text(0., 1.01, '(' + string.ascii_uppercase[7] + ')', fontsize=20, fontweight='bold', transform=ax.transAxes)
ax.set_title("Day 7", fontsize=16, fontweight='bold')

h = m.pcolormesh(lonsH, latsH, R4.isel(time=6).MAE.values, shading='gouraud', 
                 latlon=True, vmin=vmin, vmax=vmax, cmap='nipy_spectral')
m.fillcontinents(color='lightgray')
m.drawcoastlines(color='black', linewidth=1)
m.drawmeridians(np.arange(-180., 180., 0.5), labels=[0, 0, 0, 1], linewidth=0.01, fontsize=14)

# Figure 9
ax = plt.axes(list_o[8])
ax.text(0., 1.01, '(' + string.ascii_uppercase[8] + ')', fontsize=20, fontweight='bold', transform=ax.transAxes)
ax.set_title("Day 14", fontsize=16, fontweight='bold')

h = m.pcolormesh(lonsH, latsH, R4.isel(time=13).MAE.values, shading='gouraud', 
                 latlon=True, vmin=vmin, vmax=vmax, cmap='nipy_spectral')
m.fillcontinents(color='lightgray')
m.drawcoastlines(color='black', linewidth=1)
m.drawmeridians(np.arange(-180., 180., 0.5), labels=[0, 0, 0, 1], linewidth=0.01, fontsize=14)

# Figure 10
ax = plt.axes(list_o[9])
ax.text(0., 1.01, '(' + string.ascii_uppercase[9] + ')', fontsize=20, fontweight='bold', transform=ax.transAxes)
ax.set_title("Day 21", fontsize=16, fontweight='bold')

h = m.pcolormesh(lonsH, latsH, R4.isel(time=20).MAE.values, shading='gouraud', 
                 latlon=True, vmin=vmin, vmax=vmax, cmap='nipy_spectral')
m.fillcontinents(color='lightgray')
m.drawcoastlines(color='black', linewidth=1)
m.drawmeridians(np.arange(-180., 180., 0.5), labels=[0, 0, 0, 1], linewidth=0.01, fontsize=14)

# Figure 11
ax = plt.axes(list_o[10])
ax.text(0., 1.01, '(' + string.ascii_uppercase[10] + ')', fontsize=20, fontweight='bold', transform=ax.transAxes)
ax.set_title("Day 28", fontsize=16, fontweight='bold')

h = m.pcolormesh(lonsH, latsH, R4.isel(time=27).MAE.values, shading='gouraud', 
                 latlon=True, vmin=vmin, vmax=vmax, cmap='nipy_spectral')
m.fillcontinents(color='lightgray')
m.drawcoastlines(color='black', linewidth=1)
m.drawmeridians(np.arange(-180., 180., 0.5), labels=[0, 0, 0, 1], linewidth=0.01, fontsize=14)

# Colorbar
#ax_cbar = plt.axes(list_o[11])
#cbar = plt.colorbar(h, cax=ax_cbar, label='SST (°C)', orientation='vertical')
#cbar.set_label('MAE (°C)', fontsize=17)
#cbar.ax.tick_params(labelsize=17)