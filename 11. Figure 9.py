import pandas as pd 
import warnings
warnings.filterwarnings("ignore")
# Region 1
R1 = pd.read_csv("./raw_data/Events_Upwelling_R1.csv", parse_dates=["date_start"])
# Region 4
R4 = pd.read_csv("./raw_data/Events_Upwelling_R4.csv",parse_dates=["date_start"])

# Mean Intensity 
# Region 1
R1_I= R1[["date_start","int_mean"]]
R1_I['year']= R1_I['date_start'].dt.year
R1_I = R1_I.drop(columns=["date_start"])
R1_I = R1_I.groupby('year').mean().reset_index()
# Region 4
R4_I= R4[["date_start","int_mean"]]
R4_I['year']= R4_I['date_start'].dt.year
R4_I = R4_I.drop(columns=["date_start"])
R4_I = R4_I.groupby('year').mean().reset_index()

# Duration
R1_D= R1[["date_start","duration"]]
R1_D['year']= R1_D['date_start'].dt.year
R1_D = R1_D.drop(columns=["date_start"])
R1_D = R1_D.groupby('year').mean().reset_index()
# Region 4
R4_D= R4[["date_start","duration"]]
R4_D['year']= R4_D['date_start'].dt.year
R4_D = R4_D.drop(columns=["date_start"])
R4_D = R4_D.groupby('year').mean().reset_index()

R1_F= R1[["date_start","event_no"]]
R1_F['year']= R1_F['date_start'].dt.year
R1_F = R1_F.drop(columns=["date_start"])
R1_F = R1_F.groupby('year').size().reset_index().rename(columns={0:"frequency"})
# Region 4
R4_F= R4[["date_start","event_no"]]
R4_F['year']= R4_F['date_start'].dt.year
R4_F = R4_F.drop(columns=["date_start"])
R4_F = R4_F.groupby('year').size().reset_index().rename(columns={0:"frequency"})

new_row = pd.DataFrame({'year': [2023], 'frequency': [0]})
R1_F =  pd.concat([R1_F, new_row], ignore_index=True)

import matplotlib.pyplot as plt
fig,ax = plt.subplots(3,1,figsize=(15,8),sharex=True)
ax[0].plot(R1_I['year'],R1_I['int_mean'],label="", marker='o', markersize=5, color='darkblue')  
ax[0].plot(R4_I['year'],R4_I['int_mean'],label="",marker='o', markersize=5, color='darkred')  
ax[0].set_title("Mean Intensity", fontsize=15, fontweight='bold')
ax[0].set_ylabel("°C", fontsize=15, fontweight='bold')
ax[0].text(-0.01, 1.12, '(A)', transform=ax[0].transAxes, fontsize=16, fontweight='bold', va='top', ha='right')
ax[0].text(0.3, 0.5, '$0.07 \pm 0.04, p<0.1*$', transform=ax[0].transAxes, fontsize=14, fontweight='bold', va='top', ha='right', color='darkred')
ax[0].text(0.6, 0.92, '$0.05 \pm 0.03, p<0.1*$', transform=ax[0].transAxes, fontsize=14, fontweight='bold', va='top', ha='right', color='darkblue')

ax[1].plot(R1_D['year'],R1_D['duration'],label="", marker='o', markersize=5, color='darkblue')
ax[1].plot(R4_D['year'],R4_D['duration'],label="",marker='o', markersize=5, color='darkred')
ax[1].set_title("Duration", fontsize=15, fontweight='bold')
ax[1].set_ylabel("Days", fontsize=15, fontweight='bold')
ax[1].text(-0.01, 1.12, '(B)', transform=ax[1].transAxes, fontsize=16, fontweight='bold', va='top', ha='right')
ax[1].text(0.25, 0.5, '$-0.39 \pm 1.99, p>0.1$', transform=ax[1].transAxes, fontsize=14, fontweight='bold', va='top', ha='right', color='darkred')
ax[1].text(0.63, 0.87, '$-2.60 \pm 2.87, p>0.1$', transform=ax[1].transAxes, fontsize=14, fontweight='bold', va='top', ha='right', color='darkblue')

ax[2].plot(R1_F['year'],R1_F['frequency'],label="Region 1", marker='o', markersize=5, color='darkblue')
ax[2].plot(R4_F['year'],R4_F['frequency'],label="Region 4",marker='o', markersize=5, color='darkred')
ax[2].set_title("Frequency", fontsize=15, fontweight='bold')
ax[2].set_ylabel("Events", fontsize=15, fontweight='bold')
ax[2].set_xlabel("Year", fontsize=15, fontweight='bold')
ax[2].text(-0.01, 1.16, '(C)', transform=ax[2].transAxes, fontsize=16, fontweight='bold', va='top', ha='right')
ax[2].text(0.25, 0.6, '$-0.20 \pm 1.18, p>0.1$', transform=ax[2].transAxes, fontsize=14, fontweight='bold', va='top', ha='right', color='darkred')
ax[2].text(0.55, 0.87, '$0.16 \pm 1.14, p>0.1$', transform=ax[2].transAxes, fontsize=14, fontweight='bold', va='top', ha='right', color='darkblue')
ax[2].set_yticks([0, 2, 4, 6, 8, 10, 12])

# Add grid to all subplots
for a in ax:
    a.grid(which='major', linestyle='--', linewidth=0.5)
    a.minorticks_on()  # Enable minor ticks
    #a.grid(which='minor', linestyle=':', linewidth=0.5)

# Collect all handles and labels for the legend
handles, labels = [], []
for a in ax:
    h, l = a.get_legend_handles_labels()
    handles.extend(h)
    labels.extend(l)

# Place the legend below the plot in the center outside
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.09), ncol=4, fontsize=12)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.subplots_adjust(top=0.93, bottom=0.15, hspace=0.3)