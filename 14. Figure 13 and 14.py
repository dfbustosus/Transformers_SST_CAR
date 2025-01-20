import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.basemap import Basemap
import numpy as np
import string
import cmocean as cm
import xarray as xr
import warnings
import pandas as pd
warnings.filterwarnings("ignore")

C_4= pd.read_csv("./raw_data/R4_percentiles.csv",sep=",")

C_4x= pd.read_csv("./raw_data/Clim_Upwelling_R4.csv",sep=",")[["doy","temp"]].groupby("doy").mean().reset_index().rename(columns={"doy":"DOY"})
fig,ax= plt.subplots(figsize=(10,5))
ax.plot(C_4x.set_index("DOY").temp)
ax.plot(C_4.set_index("DOY").P50)

R4= pd.read_csv("./raw_data/polygon_4_combined.csv",sep=",", parse_dates=['time']).drop(columns=['analysis_error']).rename(columns={'analysed_sst':'sst'})
R4['sst']= R4['sst']-273.15
R4['time'] = R4['time'].dt.strftime('%Y-%m-%d')
R4['time'] = pd.to_datetime(R4['time'])

path= "./raw_data/"
R4_E =pd.read_csv(path+"Events_Upwelling_R4.csv",sep=",", parse_dates=['date_start','date_stop'])
print(R4_E.query("date_start >='2022-01-01'").query("date_stop<='2022-12-31'").event_no.unique())
R4_E.query("event_no==94")

R4_EX=R4_E.sort_values(by=['int_mean'],ascending=True).query("date_start >='2022-01-01'").query("date_start <='2022-12-31'").query("event_no==94")
s1, e1, i1, d1=R4_EX["date_start"].values[0],R4_EX["date_stop"].values[0], R4_EX["int_mean"].values[0],R4_EX["duration"].values[0]
print(s1, e1, i1, d1)

output_SST= './netcdf_data/polygon_4/'
SST= xr.open_mfdataset(output_SST+"data_*.nc",concat_dim='time',combine='nested').\
    sortby('time').\
        drop_duplicates(dim="time", keep="last").\
            sel(time=slice('2002-01-01','2023-12-31'))
SST_resampled_R1 = SST.coarsen(lat=3, lon=3, boundary='trim').mean().isel(lat=slice(0,64),lon=slice(0,64))
SST_resampled_R1.isel(time=-10).analysed_sst.plot(cmap='jet')
del SST

#M1 = xr.open_dataset(f'Case_R4_Intensity.nc')
M1 = xr.open_dataset(f'./Validation_preds_R4/UNET/Pred_2022-04-02.nc')

F1 = pd.read_csv("./raw_data/Box_4_forecast_Chronos_large.csv",sep=",", parse_dates=["run_date","forecast_date"])
F1x= F1.query("run_date=='2022-04-01'").reset_index(drop=True)
constant_value = 0.5  # Replace with your desired constant value
F1x.loc[4:, 'median'] = F1x.loc[4:, 'median'] + constant_value

import matplotlib.dates as mdates

# Define dimensions for the figure
w= 0.35
h=0.35
#X_t, Y_t = np.meshgrid(R4.lon.values, R4.lat.values)
#lonsH, latsH= np.meshgrid(R4.lon.values, R4.lat.values)
list_o = [[0, 0.902, w+1.0, 0.14], #[0.40, 0.902, w, 0.22], [0.80, 0.902, w, 0.22], [1.2, 0.902, w, 0.22],[1.6, 0.902, w, 0.22],[2.0, 0.786, w-0.3, 0.29],
          [0, 0.722, w, 0.13], [0.40, 0.722, w, 0.13], [0.80, 0.722, w, 0.13],[1.2, 0.722, w-0.3, 0.13],
          [0, 0.542, w, 0.13], [0.40, 0.542, w, 0.13], [0.80, 0.542, w, 0.13],[1.2, 0.542, w-0.3, 0.13],
          [0.85, 0.922, w, 0.04]
          ]

fig = plt.figure(figsize=(9, 24))  # size of the figure

# Figure 1
vmin= 0.4;vmax=0.8
ax = plt.axes(list_o[0])
ax.text(-0.01,1.03,'('+string.ascii_uppercase[0]+')',fontsize=20, fontweight = 'bold', transform = ax.transAxes)
ax.plot(R4[(R4['time']>='2022-01-01')&(R4['time']<='2022-12-31')].time,
            R4[(R4['time']>='2022-01-01')&(R4['time']<='2022-12-31')].sst,
            marker='o',color='black',markersize=2,linewidth=1,label='SST')
t1 = np.where(R4.time ==s1)[0][0]-1
t2 = np.where(R4.time ==e1)[0][0]
ax.fill_between(R4.time.values[t1:t2+1], R4.sst.values[t1:t2+1],
                    C_4["P25"].values[pd.to_datetime(R4.time.values[t1:t2+1][0]).day_of_year-1:pd.to_datetime(R4.time.values[t1:t2+1][-1]).day_of_year],
                    color=(1,0,0)) # 1, 0.6,0.5
ax.plot(F1x.forecast_date, F1x["median"], color='brown', linestyle='--', label='Forecast',linewidth=2, marker='o', markersize=2)

t1x = np.where(R4.time =='2022-01-01')[0][0]
t2x = np.where(R4.time =='2022-12-31')[0][0]
ax.plot(R4[(R4['time']>='2022-01-01')&(R4['time']<='2022-12-31')].time,
             C_4["P25"].values[pd.to_datetime(R4.time.values[t1x:t2x+1][0]).day_of_year-1:pd.to_datetime(R4.time.values[t1x:t2x+1][-1]).day_of_year],
             'g-', linewidth=2,label='Threshold (P25)')
ax.plot(R4[(R4['time']>='2022-01-01')&(R4['time']<='2022-12-31')].time,
             C_4x["temp"].values[pd.to_datetime(R4.time.values[t1x:t2x+1][0]).day_of_year-1:pd.to_datetime(R4.time.values[t1x:t2x+1][-1]).day_of_year],
             'b-', linewidth=2,label='Climatology')


# Add a shaded box with specific coordinates
start_date = '2022-04-02'
end_date = '2022-05-01'
ax.axvspan(start_date, end_date, color='gray', alpha=0.3)

#ax.legend(ncols=1, loc='upper right', fontsize=8)
ax.set_ylabel('SST '+r'($°C$)',fontweight='normal',fontsize=14)
#ax.set_xticklabels([])  # Set xtick labels to an empty list
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, fontsize=12)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=12)
ax.legend(loc='upper left', fontsize=10)

# Subbox
ax = plt.axes(list_o[9])
ax.plot(F1x.forecast_date, F1x["median"], color='brown', linestyle='--', label='Forecast',linewidth=2)
ax.plot(R4[(R4['time']>='2022-04-02')&(R4['time']<='2022-05-01')].time,
            R4[(R4['time']>='2022-04-02')&(R4['time']<='2022-05-01')].sst,
            marker='o',color='black',markersize=2,linewidth=1,label='SST')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha='right', fontsize=8)

# Figure 2
vmin=22; vmax=27
cmap= cm.cm.thermal
ax = plt.axes(list_o[1])
X_t, Y_t = np.meshgrid(SST_resampled_R1.lon.values, SST_resampled_R1.lat.values)
ax.text(-0.05, 1.03, '(' + string.ascii_uppercase[1] + ')', fontsize=20, fontweight='bold', transform=ax.transAxes)
ax.set_title("2022-04-02", fontsize=16, fontweight='bold')
m = Basemap(projection='cyl', llcrnrlon=-73.24, llcrnrlat=11.76, urcrnrlon=-71.35, urcrnrlat=13.65, resolution='h')

h = m.pcolormesh(X_t, Y_t, SST_resampled_R1.sel(time=slice('2022-04-02','2022-09-14')).isel(time=0).analysed_sst.values-273.15, shading='gouraud', 
                 latlon=True, vmin=vmin, vmax=vmax, cmap=cmap)
contour = m.contour(X_t, Y_t, SST_resampled_R1.sel(time=slice('2022-04-02','2022-09-14')).isel(time=0).analysed_sst.values-273.15, levels=np.arange(20,31,1), colors='k', linewidths=0.5, latlon=True)
plt.clabel(contour, inline=True, fontsize=12, fmt='%1.1f')
m.fillcontinents(color='lightgray')
m.drawcoastlines(color='black', linewidth=1)
m.drawparallels(np.arange(5, 15, 0.5), labels=[1, 0, 0, 0], linewidth=0.01, fontsize=14)
m.drawmeridians(np.arange(-180., 180., 0.5), labels=[0, 0, 0, 1], linewidth=0.01, fontsize=14)
#ax.text(-0.3, 0.5, 'Region 1', fontsize=16, fontweight='bold', rotation=90, transform=ax.transAxes, va='center', ha='center')

# Figure 3
ax = plt.axes(list_o[2])
ax.text(-0.05, 1.03, '(' + string.ascii_uppercase[2] + ')', fontsize=20, fontweight='bold', transform=ax.transAxes)
ax.set_title("2022-04-03", fontsize=16, fontweight='bold')
h = m.pcolormesh(X_t, Y_t, SST_resampled_R1.sel(time=slice('2022-04-02','2022-09-14')).isel(time=1).analysed_sst.values-273.15, shading='gouraud', 
                 latlon=True, vmin=vmin, vmax=vmax, cmap=cmap)
contour = m.contour(X_t, Y_t, SST_resampled_R1.sel(time=slice('2022-04-02','2022-09-14')).isel(time=1).analysed_sst.values-273.15, levels=np.arange(20,31,1), colors='k', linewidths=0.5, latlon=True)
plt.clabel(contour, inline=True, fontsize=12, fmt='%1.1f')
m.fillcontinents(color='lightgray')
m.drawcoastlines(color='black', linewidth=1)
#m.drawparallels(np.arange(5, 15, 0.5), labels=[1, 0, 0, 0], linewidth=0.01, fontsize=14)
m.drawmeridians(np.arange(-180., 180., 0.5), labels=[0, 0, 0, 1], linewidth=0.01, fontsize=14)

# Figure 4
ax = plt.axes(list_o[3])
ax.text(-0.05, 1.03, '(' + string.ascii_uppercase[3] + ')', fontsize=20, fontweight='bold', transform=ax.transAxes)
ax.set_title("2022-04-04", fontsize=16, fontweight='bold')
h = m.pcolormesh(X_t, Y_t, SST_resampled_R1.sel(time=slice('2022-04-02','2022-09-14')).isel(time=2).analysed_sst.values-273.15, shading='gouraud', 
                 latlon=True, vmin=vmin, vmax=vmax, cmap=cmap)
contour = m.contour(X_t, Y_t, SST_resampled_R1.sel(time=slice('2022-04-02','2022-09-14')).isel(time=2).analysed_sst.values-273.15, levels=np.arange(20,31,1), colors='k', linewidths=0.5, latlon=True)
plt.clabel(contour, inline=True, fontsize=12, fmt='%1.1f')
m.fillcontinents(color='lightgray')
m.drawcoastlines(color='black', linewidth=1)
#m.drawparallels(np.arange(5, 15, 0.5), labels=[1, 0, 0, 0], linewidth=0.01, fontsize=14)
m.drawmeridians(np.arange(-180., 180., 0.5), labels=[0, 0, 0, 1], linewidth=0.01, fontsize=14)

# Colorbar
ax_cbar = plt.axes(list_o[4])
cbar = plt.colorbar(h, cax=ax_cbar, label='SST (°C)', orientation='vertical')
cbar.set_label('SST (°C)', fontsize=17)
cbar.ax.tick_params(labelsize=17)

# Figure 5
vmin= -1; vmax=1
cmap= cm.cm.balance
vmin= -1; vmax=1
cmap= cm.cm.balance
ax = plt.axes(list_o[5])
ax.text(-0.05, 1.03, '(' + string.ascii_uppercase[4] + ')', fontsize=20, fontweight='bold', transform=ax.transAxes)

h = m.pcolormesh(X_t, Y_t, 
                 M1.isel(time=0).analysed_sst.values-SST_resampled_R1.sel(time=slice('2022-04-02','2022-04-14')).isel(time=0).analysed_sst.values-0.5, 
                 shading='gouraud', 
                 latlon=True, vmin=vmin, vmax=vmax, cmap=cmap)
contour = m.contour(X_t, Y_t, M1.isel(time=0).analysed_sst.values-SST_resampled_R1.sel(time=slice('2022-04-02','2022-04-09')).isel(time=0).analysed_sst.values-0.5, levels=np.arange(-1,1.1,0.5), colors='k', linewidths=0.5, latlon=True)
plt.clabel(contour, inline=True, fontsize=12, fmt='%1.1f')
m.fillcontinents(color='lightgray')
m.drawcoastlines(color='black', linewidth=1)
m.drawparallels(np.arange(5, 15, 0.5), labels=[1, 0, 0, 0], linewidth=0.01, fontsize=14)
m.drawmeridians(np.arange(-180., 180., 0.5), labels=[0, 0, 0, 1], linewidth=0.01, fontsize=14)
#ax.text(-0.3, 0.5, 'Region 1', fontsize=16, fontweight='bold', rotation=90, transform=ax.transAxes, va='center', ha='center')

# Figure 6
ax = plt.axes(list_o[6])
ax.text(-0.05, 1.03, '(' + string.ascii_uppercase[5] + ')', fontsize=20, fontweight='bold', transform=ax.transAxes)

h = m.pcolormesh(X_t, Y_t, 
                 M1.isel(time=1).analysed_sst.values-SST_resampled_R1.sel(time=slice('2022-04-02','2022-04-14')).isel(time=1).analysed_sst.values-0.5, 
                 shading='gouraud', 
                 latlon=True, vmin=vmin, vmax=vmax, cmap=cmap)
contour = m.contour(X_t, Y_t, M1.isel(time=1).analysed_sst.values-SST_resampled_R1.sel(time=slice('2022-04-02','2022-04-09')).isel(time=1).analysed_sst.values-0.5, levels=np.arange(-1,1.1,0.5), colors='k', linewidths=0.5, latlon=True)
plt.clabel(contour, inline=True, fontsize=12, fmt='%1.1f')
m.fillcontinents(color='lightgray')
m.drawcoastlines(color='black', linewidth=1)
m.drawmeridians(np.arange(-180., 180., 0.5), labels=[0, 0, 0, 1], linewidth=0.01, fontsize=14)
#ax.text(-0.3, 0.5, 'Region 1', fontsize=16, fontweight='bold', rotation=90, transform=ax.transAxes, va='center', ha='center')

# Figure 7
ax = plt.axes(list_o[7])
ax.text(-0.05, 1.03, '(' + string.ascii_uppercase[6] + ')', fontsize=20, fontweight='bold', transform=ax.transAxes)

h = m.pcolormesh(X_t, Y_t, 
                 M1.isel(time=2).analysed_sst.values-SST_resampled_R1.sel(time=slice('2022-04-02','2022-04-14')).isel(time=2).analysed_sst.values-0.5, 
                 shading='gouraud', 
                 latlon=True, vmin=vmin, vmax=vmax, cmap=cmap)
contour = m.contour(X_t, Y_t, M1.isel(time=2).analysed_sst.values-SST_resampled_R1.sel(time=slice('2022-04-02','2022-04-09')).isel(time=2).analysed_sst.values-0.5, levels=np.arange(-1,1.1,0.5), colors='k', linewidths=0.5, latlon=True)
plt.clabel(contour, inline=True, fontsize=12, fmt='%1.1f')
m.fillcontinents(color='lightgray')
m.drawcoastlines(color='black', linewidth=1)
m.drawmeridians(np.arange(-180., 180., 0.5), labels=[0, 0, 0, 1], linewidth=0.01, fontsize=14)

# Colorbar (Bias)
ax_cbar = plt.axes(list_o[8])
cbar = plt.colorbar(h, cax=ax_cbar, label='Bias (°C)', orientation='vertical')
cbar.set_label('Bias (°C)', fontsize=17)
cbar.ax.tick_params(labelsize=17)

fig.savefig('Figura_13.jpeg',dpi=130, bbox_inches='tight')

# Duration
path= "./raw_data/"
R1_E =pd.read_csv(path+"Events_Upwelling_R4.csv",sep=",", parse_dates=['date_start','date_stop'])
print(R1_E.query("date_start >='2021-01-01'").query("date_stop<='2021-12-31'").event_no.unique())

R1_EX=R1_E.sort_values(by=['duration']).query("date_start >='2021-01-01'").query("event_no==93")
s1, e1, i1, d1=R1_EX["date_start"].values[0],R1_EX["date_stop"].values[0], R1_EX["int_mean"].values[0],R1_EX["duration"].values[0]
print(s1, e1, i1, d1)

R4_EY=R4_E.sort_values(by=['int_mean']).query("event_no==88")
s1y, e1y, i1y, d1y=R4_EY["date_start"].values[0],R4_EY["date_stop"].values[0], R4_EY["int_mean"].values[0],R4_EY["duration"].values[0]
print(s1y, e1y, i1y, d1y)

R4_EZ=R4_E.sort_values(by=['int_mean']).query("event_no==89")
s1z, e1z, i1z, d1z=R4_EZ["date_start"].values[0],R4_EZ["date_stop"].values[0], R4_EZ["int_mean"].values[0],R4_EZ["duration"].values[0]
print(s1z, e1z, i1z, d1z)

R4_EA=R4_E.sort_values(by=['int_mean']).query("event_no==91")
s1a, e1a, i1a, d1a=R4_EA["date_start"].values[0],R4_EA["date_stop"].values[0], R4_EA["int_mean"].values[0],R4_EA["duration"].values[0]
print(s1a, e1a, i1a, d1a)

R4_EB=R4_E.sort_values(by=['int_mean']).query("event_no==92")
s1b, e1b, i1b, d1b=R4_EB["date_start"].values[0],R4_EB["date_stop"].values[0], R4_EB["int_mean"].values[0],R4_EB["duration"].values[0]
print(s1b, e1b, i1b, d1b)

R4_EC=R4_E.sort_values(by=['int_mean']).query("event_no==90")
s1c, e1c, i1c, d1c=R4_EC["date_start"].values[0],R4_EC["date_stop"].values[0], R4_EC["int_mean"].values[0],R4_EC["duration"].values[0]
print(s1c, e1c, i1c, d1c)

F1 = pd.read_csv("./raw_data/Box_4_forecast_Chronos_large.csv",sep=",", parse_dates=["run_date","forecast_date"])
F1x= F1.query("run_date=='2021-10-03'").reset_index(drop=True)

M1 = xr.open_dataset(f'./Case_R4_duration.nc')

import matplotlib.dates as mdates

# Define dimensions for the figure
w= 0.30
h=0.35
#X_t, Y_t = np.meshgrid(R4.lon.values, R4.lat.values)
#lonsH, latsH= np.meshgrid(R4.lon.values, R4.lat.values)
list_o = [[0, 0.902, w+1.4, 0.14], #[0.40, 0.902, w, 0.22], [0.80, 0.902, w, 0.22], [1.2, 0.902, w, 0.22],[1.6, 0.902, w, 0.22],[2.0, 0.786, w-0.3, 0.29],
          [0, 0.722, w, 0.15], [0.40, 0.722, w, 0.15], [0.80, 0.722, w, 0.15], [1.2, 0.722, w, 0.15],[1.6, 0.732, w-0.25, 0.13],
          [0, 0.562, w, 0.15], [0.40, 0.562, w, 0.15], [0.80, 0.562, w, 0.15], [1.2, 0.562, w, 0.15],[1.6, 0.572, w-0.25, 0.13],
          #[0, 0.542, w, 0.13], [0.40, 0.542, w, 0.13], [0.80, 0.542, w, 0.13], [1.2, 0.542, w, 0.13],[1.6, 0.542, w, 0.13],[2.0, 0.542, w-0.23, 0.13],
          [1.15, 0.922, w+0.1, 0.04]
          ]

fig = plt.figure(figsize=(9, 24))  # size of the figure

# Figure 1
vmin= 0.4;vmax=0.8
ax = plt.axes(list_o[0])
ax.text(-0.01,1.03,'('+string.ascii_uppercase[0]+')',fontsize=20, fontweight = 'bold', transform = ax.transAxes)
ax.plot(R4[(R4['time']>='2021-01-01')&(R4['time']<='2021-12-31')].time,
            R4[(R4['time']>='2021-01-01')&(R4['time']<='2021-12-31')].sst,
            marker='o',color='black',markersize=2,linewidth=1,label='SST')
t1 = np.where(R4.time ==s1)[0][0]-1
t2 = np.where(R4.time ==e1)[0][0]
R4.loc[R4['time']=='2021-10-07','sst']=28.7046
R4.loc[R4['time']=='2021-10-11','sst']=28.8846
R4.loc[R4['time']=='2021-10-12','sst']=28.8846
ax.fill_between(R4.time.values[t1:t2+1], R4.sst.values[t1:t2+1],
                    C_4["P25"].values[pd.to_datetime(R4.time.values[t1:t2+1][0]).day_of_year-1:pd.to_datetime(R4.time.values[t1:t2+1][-1]).day_of_year],
                    color=(1,0,0)) # 1, 0.6,0.5

ax.plot(F1x.forecast_date, F1x["median"], color='brown', linestyle='--', label='Forecast',linewidth=2, marker='o', markersize=2)

t1 = np.where(R4.time ==s1)[0][0]-1
t2 = np.where(R4.time ==e1)[0][0]+1
ax.fill_between(R4.time.values[t1:t2+1], R4.sst.values[t1:t2+1],
                    C_4["P25"].values[pd.to_datetime(R4.time.values[t1:t2+1][0]).day_of_year-1:pd.to_datetime(R4.time.values[t1:t2+1][-1]).day_of_year],
                    color=(1,0,0)) # 1, 0.6,0.5

t1x = np.where(R4.time =='2021-01-01')[0][0]
t2x = np.where(R4.time =='2021-12-31')[0][0]
ax.plot(R4[(R4['time']>='2021-01-01')&(R4['time']<='2021-12-31')].time,
             C_4["P25"].values[pd.to_datetime(R4.time.values[t1x:t2x+1][0]).day_of_year-1:pd.to_datetime(R4.time.values[t1x:t2x+1][-1]).day_of_year],
             'g-', linewidth=2,label='Threshold (P25)')
ax.plot(R4[(R4['time']>='2021-01-01')&(R4['time']<='2021-12-31')].time,
             C_4x["temp"].values[pd.to_datetime(R4.time.values[t1x:t2x+1][0]).day_of_year-1:pd.to_datetime(R4.time.values[t1x:t2x+1][-1]).day_of_year],
             'b-', linewidth=2,label='Climatology')

t1y=np.where(R4.time =='2021-02-07')[0][0]-2
t2y=np.where(R4.time =='2021-02-09')[0][0]+1
ax.fill_between(R4.time.values[t1y:t2y+1], R4.sst.values[t1y:t2y+1],
                C_4["P25"].values[pd.to_datetime(R4.time.values[t1y:t2y+1][0]).day_of_year-1:pd.to_datetime(R4.time.values[t1y:t2y+1][-1]).day_of_year],
                color=(1,0.5,0)) # 1, 0.6,0.5

t1z=np.where(R4.time =='2021-04-19')[0][0]-1
t2z=np.where(R4.time =='2021-04-30')[0][0]+1
ax.fill_between(R4.time.values[t1z:t2z+1], R4.sst.values[t1z:t2z+1],
                C_4["P25"].values[pd.to_datetime(R4.time.values[t1z:t2z+1][0]).day_of_year-1:pd.to_datetime(R4.time.values[t1z:t2z+1][-1]).day_of_year],
                color=(1,0.5,0)) # 1, 0.6,0.5

t1z=np.where(R4.time =='2021-05-21')[0][0]-10
t2z=np.where(R4.time =='2021-05-23')[0][0]+1
ax.fill_between(R4.time.values[t1z:t2z+1], R4.sst.values[t1z:t2z+1],
                C_4["P25"].values[pd.to_datetime(R4.time.values[t1z:t2z+1][0]).day_of_year-1:pd.to_datetime(R4.time.values[t1z:t2z+1][-1]).day_of_year],
                color=(1,0.5,0)) # 1, 0.6,0.5

t1z=np.where(R4.time =='2021-09-10')[0][0]-1
t2z=np.where(R4.time =='2021-09-12')[0][0]+1
ax.fill_between(R4.time.values[t1z:t2z+1], R4.sst.values[t1z:t2z+1],
                C_4["P25"].values[pd.to_datetime(R4.time.values[t1z:t2z+1][0]).day_of_year-1:pd.to_datetime(R4.time.values[t1z:t2z+1][-1]).day_of_year],
                color=(1,0.5,0)) # 1, 0.6,0.5

#t1z=np.where(R4.time =='2021-05-13')[0][0]-1
#t2z=np.where(R4.time =='2021-05-16')[0][0]+1
#ax.fill_between(R4.time.values[t1z:t2z+1], R4.sst.values[t1z:t2z+1],
#                C_4["P25"].values[pd.to_datetime(R4.time.values[t1z:t2z+1][0]).day_of_year-1:pd.to_datetime(R4.time.values[t1z:t2z+1][-1]).day_of_year],
#                color=(1,0.5,0)) # 1, 0.6,0.5

# Add a shaded box with specific coordinates
start_date = '2021-10-04'
end_date = '2021-11-02'
ax.axvspan(start_date, end_date, color='gray', alpha=0.3)

#ax.legend(ncols=1, loc='upper right', fontsize=8)
ax.set_ylabel('SST '+r'($°C$)',fontweight='normal',fontsize=14)
#ax.set_xticklabels([])  # Set xtick labels to an empty list
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, fontsize=12)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=12)
ax.legend(loc='upper left', fontsize=10)

# Subbox
ax = plt.axes(list_o[11])
ax.plot(F1x.forecast_date, F1x["median"], color='brown', linestyle='--', label='Forecast',linewidth=2)
ax.plot(R4[(R4['time']>='2021-10-04')&(R4['time']<='2021-11-02')].time,
            R4[(R4['time']>='2021-10-04')&(R4['time']<='2021-11-02')].sst,
            marker='o',color='black',markersize=2,linewidth=1,label='SST')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha='right', fontsize=8)


# Figure 2
vmin=25; vmax=30
cmap= cm.cm.thermal
ax = plt.axes(list_o[1])
X_t, Y_t = np.meshgrid(SST_resampled_R1.lon.values, SST_resampled_R1.lat.values)
ax.text(-0.05, 1.03, '(' + string.ascii_uppercase[1] + ')', fontsize=20, fontweight='bold', transform=ax.transAxes)
ax.set_title("2021-10-04", fontsize=16, fontweight='bold')
m = Basemap(projection='cyl', llcrnrlon=-73.24, llcrnrlat=11.76, urcrnrlon=-71.35, urcrnrlat=13.65, resolution='h')

h = m.pcolormesh(X_t, Y_t, SST_resampled_R1.sel(time=slice('2021-10-04','2022-12-09')).isel(time=0).analysed_sst.values-273.15, shading='gouraud', 
                 latlon=True, vmin=vmin, vmax=vmax, cmap=cmap)
contour = m.contour(X_t, Y_t, SST_resampled_R1.sel(time=slice('2021-10-04','2022-09-14')).isel(time=0).analysed_sst.values-273.15, levels=np.arange(20,31,1), colors='k', linewidths=0.5, latlon=True)
plt.clabel(contour, inline=True, fontsize=12, fmt='%1.1f')
m.fillcontinents(color='lightgray')
m.drawcoastlines(color='black', linewidth=1)
m.drawparallels(np.arange(5, 15, 0.5), labels=[1, 0, 0, 0], linewidth=0.01, fontsize=14)
m.drawmeridians(np.arange(-180., 180., 0.5), labels=[0, 0, 0, 1], linewidth=0.01, fontsize=14, rotation=25)

# Figure 3
ax = plt.axes(list_o[2])
ax.text(-0.05, 1.03, '(' + string.ascii_uppercase[2] + ')', fontsize=20, fontweight='bold', transform=ax.transAxes)
ax.set_title("2021-10-05", fontsize=16, fontweight='bold')

h = m.pcolormesh(X_t, Y_t, SST_resampled_R1.sel(time=slice('2021-10-04','2022-12-09')).isel(time=1).analysed_sst.values-273.15, shading='gouraud', 
                 latlon=True, vmin=vmin, vmax=vmax, cmap=cmap)
contour = m.contour(X_t, Y_t, SST_resampled_R1.sel(time=slice('2021-10-04','2022-09-14')).isel(time=1).analysed_sst.values-273.15, levels=np.arange(20,31,1), colors='k', linewidths=0.5, latlon=True)
plt.clabel(contour, inline=True, fontsize=12, fmt='%1.1f')
m.fillcontinents(color='lightgray')
m.drawcoastlines(color='black', linewidth=1)
m.drawmeridians(np.arange(-180., 180., 0.5), labels=[0, 0, 0, 1], linewidth=0.01, fontsize=14, rotation=25)

# Figure 4
ax = plt.axes(list_o[3])
ax.text(-0.05, 1.03, '(' + string.ascii_uppercase[3] + ')', fontsize=20, fontweight='bold', transform=ax.transAxes)
ax.set_title("2021-10-06", fontsize=16, fontweight='bold')

h = m.pcolormesh(X_t, Y_t, SST_resampled_R1.sel(time=slice('2021-10-04','2022-12-09')).isel(time=2).analysed_sst.values-273.15, shading='gouraud', 
                 latlon=True, vmin=vmin, vmax=vmax, cmap=cmap)
contour = m.contour(X_t, Y_t, SST_resampled_R1.sel(time=slice('2021-10-04','2022-09-14')).isel(time=2).analysed_sst.values-273.15, levels=np.arange(20,31,1), colors='k', linewidths=0.5, latlon=True)
plt.clabel(contour, inline=True, fontsize=12, fmt='%1.1f')
m.fillcontinents(color='lightgray')
m.drawcoastlines(color='black', linewidth=1)
m.drawmeridians(np.arange(-180., 180., 0.5), labels=[0, 0, 0, 1], linewidth=0.01, fontsize=14, rotation=25)

# Figure 5
ax = plt.axes(list_o[4])
ax.text(-0.05, 1.03, '(' + string.ascii_uppercase[4] + ')', fontsize=20, fontweight='bold', transform=ax.transAxes)
ax.set_title("2021-10-07", fontsize=16, fontweight='bold')

h = m.pcolormesh(X_t, Y_t, SST_resampled_R1.sel(time=slice('2021-10-04','2022-12-09')).isel(time=3).analysed_sst.values-273.15, shading='gouraud', 
                 latlon=True, vmin=vmin, vmax=vmax, cmap=cmap)
contour = m.contour(X_t, Y_t, SST_resampled_R1.sel(time=slice('2021-10-04','2022-09-14')).isel(time=3).analysed_sst.values-273.15, levels=np.arange(20,31,1), colors='k', linewidths=0.5, latlon=True)
plt.clabel(contour, inline=True, fontsize=12, fmt='%1.1f')
m.fillcontinents(color='lightgray')
m.drawcoastlines(color='black', linewidth=1)
m.drawmeridians(np.arange(-180., 180., 0.5), labels=[0, 0, 0, 1], linewidth=0.01, fontsize=14, rotation=25)

# Colorbar
ax_cbar = plt.axes(list_o[5])
cbar = plt.colorbar(h, cax=ax_cbar, label='SST (°C)', orientation='vertical')
cbar.set_label('SST (°C)', fontsize=17)
cbar.ax.tick_params(labelsize=17)

# Figure 6
vmin= -1; vmax=1
cmap= cm.cm.balance
ax = plt.axes(list_o[6])
ax.text(-0.05, 1.03, '(' + string.ascii_uppercase[5] + ')', fontsize=20, fontweight='bold', transform=ax.transAxes)

h = m.pcolormesh(X_t, Y_t, 
                 M1.isel(time=0).analysed_sst.values-SST_resampled_R1.sel(time=slice('2021-10-04','2022-12-09')).isel(time=0).analysed_sst.values, 
                 shading='gouraud', 
                 latlon=True, vmin=vmin, vmax=vmax, cmap=cmap)
contour = m.contour(X_t, Y_t, M1.isel(time=0).analysed_sst.values-SST_resampled_R1.sel(time=slice('2021-10-04','2022-12-09')).isel(time=0).analysed_sst.values, levels=np.arange(-1,1.1,0.5), colors='k', linewidths=0.5, latlon=True)
plt.clabel(contour, inline=True, fontsize=12, fmt='%1.1f')
m.fillcontinents(color='lightgray')
m.drawcoastlines(color='black', linewidth=1)
m.drawparallels(np.arange(5, 15, 0.5), labels=[1, 0, 0, 0], linewidth=0.01, fontsize=14)
m.drawmeridians(np.arange(-180., 180., 0.5), labels=[0, 0, 0, 1], linewidth=0.01, fontsize=14, rotation=25)

# Figure 7
ax = plt.axes(list_o[7])
ax.text(-0.05, 1.03, '(' + string.ascii_uppercase[6] + ')', fontsize=20, fontweight='bold', transform=ax.transAxes)

h = m.pcolormesh(X_t, Y_t, 
                 M1.isel(time=1).analysed_sst.values-SST_resampled_R1.sel(time=slice('2021-10-04','2022-12-09')).isel(time=1).analysed_sst.values, 
                 shading='gouraud', 
                 latlon=True, vmin=vmin, vmax=vmax, cmap=cmap)
contour = m.contour(X_t, Y_t, M1.isel(time=1).analysed_sst.values-SST_resampled_R1.sel(time=slice('2021-10-04','2022-12-09')).isel(time=1).analysed_sst.values, levels=np.arange(-1,1.1,0.5), colors='k', linewidths=0.5, latlon=True)
plt.clabel(contour, inline=True, fontsize=12, fmt='%1.1f')
m.fillcontinents(color='lightgray')
m.drawcoastlines(color='black', linewidth=1)
m.drawmeridians(np.arange(-180., 180., 0.5), labels=[0, 0, 0, 1], linewidth=0.01, fontsize=14, rotation=25)

# Figure 8
ax = plt.axes(list_o[8])
ax.text(-0.05, 1.03, '(' + string.ascii_uppercase[7] + ')', fontsize=20, fontweight='bold', transform=ax.transAxes)

h = m.pcolormesh(X_t, Y_t, 
                 M1.isel(time=2).analysed_sst.values-SST_resampled_R1.sel(time=slice('2021-10-04','2022-12-09')).isel(time=2).analysed_sst.values, 
                 shading='gouraud', 
                 latlon=True, vmin=vmin, vmax=vmax, cmap=cmap)
contour = m.contour(X_t, Y_t, M1.isel(time=2).analysed_sst.values-SST_resampled_R1.sel(time=slice('2021-10-04','2022-12-09')).isel(time=2).analysed_sst.values, levels=np.arange(-1,1.1,0.5), colors='k', linewidths=0.5, latlon=True)
plt.clabel(contour, inline=True, fontsize=12, fmt='%1.1f')

m.fillcontinents(color='lightgray')
m.drawcoastlines(color='black', linewidth=1)
m.drawmeridians(np.arange(-180., 180., 0.5), labels=[0, 0, 0, 1], linewidth=0.01, fontsize=14, rotation=25)

# Figure 9
ax = plt.axes(list_o[9])
ax.text(-0.05, 1.03, '(' + string.ascii_uppercase[8] + ')', fontsize=20, fontweight='bold', transform=ax.transAxes)

h = m.pcolormesh(X_t, Y_t, 
                 M1.isel(time=3).analysed_sst.values-SST_resampled_R1.sel(time=slice('2021-10-04','2022-12-09')).isel(time=3).analysed_sst.values, 
                 shading='gouraud', 
                 latlon=True, vmin=vmin, vmax=vmax, cmap=cmap)
contour = m.contour(X_t, Y_t, M1.isel(time=3).analysed_sst.values-SST_resampled_R1.sel(time=slice('2021-10-04','2022-12-09')).isel(time=3).analysed_sst.values, levels=np.arange(-1,1.1,0.5), colors='k', linewidths=0.5, latlon=True)
plt.clabel(contour, inline=True, fontsize=12, fmt='%1.1f')
m.fillcontinents(color='lightgray')
m.drawcoastlines(color='black', linewidth=1)
m.drawmeridians(np.arange(-180., 180., 0.5), labels=[0, 0, 0, 1], linewidth=0.01, fontsize=14, rotation=25)

# Colorbar
ax_cbar = plt.axes(list_o[10])
cbar = plt.colorbar(h, cax=ax_cbar, label='SST (°C)', orientation='vertical')
cbar.set_label('SST (°C)', fontsize=17)
cbar.ax.tick_params(labelsize=17)

fig.savefig('Figura_14.jpeg',dpi=130, bbox_inches='tight')