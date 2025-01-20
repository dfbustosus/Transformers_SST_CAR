import pandas as pd 
import xarray as xr 
import numpy as np
import matplotlib.pyplot as plt

path="./raw_data/polygon_4_combined.csv"
df= pd.read_csv(path, parse_dates=["time"])
df['t'] = df['time'].dt.strftime('%Y-%m-%d')
df["t"]= pd.to_datetime(df["t"])
df= df.assign(temp= df["analysed_sst"]-273.15).drop(columns=["analysed_sst","analysis_error","time"])

data_clim= df

start_year = 2003  # Start year of the climatological base period
end_year = 2023  # End year of the climatological base period
base_period_start = 1  # Start day of the climatological base period (December 26)
base_period_end = 365  # End day of the climatological base period (December 31)

climatological_p25= np.zeros(base_period_end - base_period_start + 1)
for j in range(base_period_start, base_period_end + 1):
    temperatures = []
    for year in range(start_year, end_year + 1):
        for d in range(j - 5, j + 6):
            if d<=0 or d >365:
                a='Nada'
            else:
                daily_temperatures = data_clim[(data_clim['t'].dt.year == year) & (data_clim['t'].dt.dayofyear == d)]
                temperatures.append(daily_temperatures['temp'].values[0]) 
    p25=np.nanpercentile(np.array(temperatures),25)
    climatological_p25[j - 1] = p25
    print('Dia clim: ', j, '----------', p25,'---------------- Ok')


start_year = 2003  # Start year of the climatological base period
end_year = 2023  # End year of the climatological base period
base_period_start = 1  # Start day of the climatological base period (December 26)
base_period_end = 365  # End day of the climatological base period (December 31)

climatological_p50= np.zeros(base_period_end - base_period_start + 1)
for j in range(base_period_start, base_period_end + 1):
    temperatures = []
    for year in range(start_year, end_year + 1):
        for d in range(j - 5, j + 6):
            if d<=0 or d >365:
                a='Nada'
            else:
                daily_temperatures = data_clim[(data_clim['t'].dt.year == year) & (data_clim['t'].dt.dayofyear == d)]
                temperatures.append(daily_temperatures['temp'].values[0]) 
    p50=np.nanpercentile(np.array(temperatures),50)
    climatological_p50[j - 1] = p50
    print('Dia clim: ', j, '----------', p50,'---------------- Ok')

R4= pd.DataFrame()
R4["DOY"]= np.arange(1,366)
R4['P25']= climatological_p25
R4['P50']= climatological_p50
R4.to_csv('./raw_data/R4_Percentiles.csv', index=False)

path="./raw_data/polygon_3_combined.csv"
df= pd.read_csv(path, parse_dates=["time"])
df['t'] = df['time'].dt.strftime('%Y-%m-%d')
df["t"]= pd.to_datetime(df["t"])
df= df.assign(temp= df["analysed_sst"]-273.15).drop(columns=["analysed_sst","analysis_error","time"])
data_clim= df


start_year = 2003  # Start year of the climatological base period
end_year = 2023  # End year of the climatological base period
base_period_start = 1  # Start day of the climatological base period (December 26)
base_period_end = 365  # End day of the climatological base period (December 31)

climatological_p25= np.zeros(base_period_end - base_period_start + 1)
for j in range(base_period_start, base_period_end + 1):
    temperatures = []
    for year in range(start_year, end_year + 1):
        for d in range(j - 5, j + 6):
            if d<=0 or d >365:
                a='Nada'
            else:
                daily_temperatures = data_clim[(data_clim['t'].dt.year == year) & (data_clim['t'].dt.dayofyear == d)]
                temperatures.append(daily_temperatures['temp'].values[0]) 
    p25=np.nanpercentile(np.array(temperatures),25)
    climatological_p25[j - 1] = p25
    print('Dia clim: ', j, '----------', p25,'---------------- Ok')


start_year = 2003  # Start year of the climatological base period
end_year = 2023  # End year of the climatological base period
base_period_start = 1  # Start day of the climatological base period (December 26)
base_period_end = 365  # End day of the climatological base period (December 31)

climatological_p50= np.zeros(base_period_end - base_period_start + 1)
for j in range(base_period_start, base_period_end + 1):
    temperatures = []
    for year in range(start_year, end_year + 1):
        for d in range(j - 5, j + 6):
            if d<=0 or d >365:
                a='Nada'
            else:
                daily_temperatures = data_clim[(data_clim['t'].dt.year == year) & (data_clim['t'].dt.dayofyear == d)]
                temperatures.append(daily_temperatures['temp'].values[0]) 
    p50=np.nanpercentile(np.array(temperatures),50)
    climatological_p50[j - 1] = p50
    print('Dia clim: ', j, '----------', p50,'---------------- Ok')

R1= pd.DataFrame()
R1["DOY"]= np.arange(1,366)
R1['P25']= climatological_p25
R1['P50']= climatological_p50
R1.to_csv('./raw_data/R1_Percentiles.csv', index=False)


R1= pd.DataFrame()
R1["DOY"]= np.arange(1,366)
R1['P25']= climatological_p25
R1['P50']= climatological_p50
R1.to_csv('./raw_data/R1_Percentiles.csv', index=False)

import matplotlib.dates as mdates
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 2, figsize=(16, 8), sharex=True, sharey=True)

# Plot data for Region 1
ax[0].plot(R1['DOY'], R1['P50'], label='')
ax[0].plot(R1['DOY'], R1['P25'], label='')
ax[0].set_ylabel('SST (°C)', fontsize=12)
ax[0].set_xlabel('Day of year', fontsize=12)
ax[0].set_title('Region 1', fontweight='bold', fontsize=14)
ax[0].text(0.01, 1.03, '(A)', transform=ax[0].transAxes, fontsize=16, fontweight='bold', va='top', ha='right')

# Plot data for Region 4
ax[1].plot(R4['DOY'], R4['P50'], label='P50')
ax[1].plot(R4['DOY'], R4['P25'], label='P25')
ax[1].set_xlabel('Day of year', fontsize=12)
ax[1].set_title('Region 4', fontweight='bold', fontsize=14)
ax[1].text(0.01, 1.03, '(B)', transform=ax[1].transAxes, fontsize=16, fontweight='bold', va='top', ha='right')

# Set major and minor locators for the x-axis
ax[0].xaxis.set_major_locator(mdates.MonthLocator())
ax[0].xaxis.set_major_formatter(mdates.DateFormatter('%b'))
ax[0].xaxis.set_minor_locator(mdates.DayLocator(interval=10))

ax[1].xaxis.set_major_locator(mdates.MonthLocator())
ax[1].xaxis.set_major_formatter(mdates.DateFormatter('%b'))
ax[1].xaxis.set_minor_locator(mdates.DayLocator(interval=10))

# Add grid to both subplots for major ticks only
for a in ax:
    a.grid(which='major', linestyle='--', linewidth=0.5)
    a.minorticks_on()  # Enable minor ticks

handles, labels = [], []
for a in ax:
    h, l = a.get_legend_handles_labels()
    handles.extend(h)
    labels.extend(l)

# Place the legend below the plot in the center outside
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.92, 0.15), ncol=4, fontsize=12)

# Adjust layout
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.subplots_adjust(top=0.93, bottom=0.1, hspace=0.3, wspace=0.2)

# Save the figure as a high-resolution JPEG without whitespace
fig.savefig('Figure_5_final.jpeg', format='jpeg', dpi=300, bbox_inches='tight')

plt.show()