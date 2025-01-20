import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

ruta= "./raw_data/"
file_pattern = ruta+'Events_Upwelling_R1.csv'
R1_I = pd.read_csv(file_pattern, parse_dates=["date_start"])
R1_I['month'] = R1_I['date_start'].dt.month
R1_I['year'] = R1_I['date_start'].dt.year
R1_I= R1_I[["month","year","int_mean","duration"]]

grouped_R1_I = R1_I.groupby(['month', 'year']).agg(
    int_mean=('int_mean', 'mean'),
    duration_mean=('duration', 'mean'),
    event_count=('int_mean', 'size')
).reset_index()
grouped_R1_I.sort_values(by=['year','month'],inplace=True)

file_pattern = ruta+'Events_Upwelling_R4.csv'
R4_I = pd.read_csv(file_pattern, parse_dates=["date_start"])
R4_I['month'] = R4_I['date_start'].dt.month
R4_I['year'] = R4_I['date_start'].dt.year
R4_I= R4_I[["month","year","int_mean","duration"]]

grouped_R4_I = R4_I.groupby(['month', 'year']).agg(
    int_mean=('int_mean', 'mean'),
    duration_mean=('duration', 'mean'),
    event_count=('int_mean', 'size')
).reset_index()
grouped_R4_I.sort_values(by=['year','month'],inplace=True)

import cmocean 
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.colors as mcolors
import string
cmap = plt.get_cmap('rainbow')
# Get the colors from the coolwarm colormap
colors = cmap(np.linspace(0, 1, cmap.N))
# Remove the middle color (index: cmap.N//2)
colors = np.delete(colors, cmap.N // 2, axis=0)
# Create a new colormap using the modified colors
new_cmap = mcolors.ListedColormap(colors)

fig = plt.figure(figsize=(16,12)) # figura
gs = GridSpec(4,3, width_ratios=[1, 1, 1],height_ratios=[0.15,0.3, 0.3, 0.01])
hist1=fig.add_subplot(gs[0]) # primer hist
hist2=fig.add_subplot(gs[1]) # segundo hist
hist3=fig.add_subplot(gs[2]) # tercer hist
bar_width = 0.4
# Figure 1
h1 = hist1.bar(
    x=grouped_R1_I.dropna().groupby('month').mean()['int_mean'].index - bar_width/2, 
    height=grouped_R1_I.dropna().groupby('month').mean()['int_mean'].values,
    width=bar_width, color='darkblue', label='Region 1'
)
h2 = hist1.bar(
    x=grouped_R4_I.dropna().groupby('month').mean()['int_mean'].index + bar_width/2, 
    height=grouped_R4_I.dropna().groupby('month').mean()['int_mean'].values,
    width=bar_width, color='darkred', label='Region 4'
)
hist1.text(-0.05,1.03,'('+string.ascii_uppercase[0]+')',fontsize=16, fontweight = 'bold', transform = hist1.transAxes)

# Figure 2
h1 = hist2.bar(
    x=grouped_R1_I.dropna().groupby('month').mean()['duration_mean'].index - bar_width/2, 
    height=grouped_R1_I.dropna().groupby('month').mean()['duration_mean'].values,
    width=bar_width, color='darkblue', label=''
)
h2 = hist2.bar(
    x=grouped_R4_I.dropna().groupby('month').mean()['duration_mean'].index + bar_width/2, 
    height=grouped_R4_I.dropna().groupby('month').mean()['duration_mean'].values,
    width=bar_width, color='darkred', label=''
)
hist2.text(-0.05,1.03,'('+string.ascii_uppercase[1]+')',fontsize=16, fontweight = 'bold', transform = hist2.transAxes)

# Figure 3
h1 = hist3.bar(
    x=grouped_R1_I.dropna().groupby('month').mean()['event_count'].index - bar_width/2, 
    height=grouped_R1_I.dropna().groupby('month').mean()['event_count'].values,
    width=bar_width, color='darkblue', label='Region 1'
)
h2 = hist3.bar(
    x=grouped_R4_I.dropna().groupby('month').mean()['event_count'].index + bar_width/2, 
    height=grouped_R4_I.dropna().groupby('month').mean()['event_count'].values,
    width=bar_width, color='darkred', label='Region 4'
)
hist3.text(-0.05,1.03,'('+string.ascii_uppercase[2]+')',fontsize=16, fontweight = 'bold', transform = hist3.transAxes)

hist1.set_title('Mean Intensity',fontsize=14, fontweight='bold')
hist1.set_ylabel('[°C]',fontsize=14, fontweight='bold')
hist1.set_ylim([-1.0,-0.5])
hist2.set_title('Mean Duration',fontsize=14, fontweight='bold')
hist2.set_ylabel('Events',fontsize=14, fontweight='bold')
hist2.set_ylim([5,15])
hist3.set_title('Frequency',fontsize=14, fontweight='bold')
hist3.set_ylabel('Days',fontsize=14, fontweight='bold')
hist3.set_ylim([0.7,1.7])
hist3.legend(loc='upper left',fontsize=10, ncol=2)

hist1.set_xticklabels([]);hist2.set_xticklabels([]);hist3.set_xticklabels([]);
yticks = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
yticklabels = ['Jan', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul','Aug' ,'Sep','Oct','Nov','Dec']
hist1.set_xticks(yticks);hist2.set_xticks(yticks);hist3.set_xticks(yticks)

# Figure 4, 5, 6
years= np.arange(2002, 2024,1)
ax1 = fig.add_subplot(gs[3]) # primer heatmsp 
ax2 = fig.add_subplot(gs[4]) # segundo heatmap
ax3 = fig.add_subplot(gs[5]) # segundo heatmap

scatter1 = ax1.scatter(grouped_R1_I.month, grouped_R1_I.year, c=grouped_R1_I.int_mean, cmap=cmap, marker='s', vmin=-1.3, vmax=-0.6,s=70)
ax1.text(-0.05,1.01,'('+string.ascii_uppercase[3]+')',fontsize=16, fontweight = 'bold', transform = ax1.transAxes)
ax1.set_ylabel('Region 1',fontsize=14, fontweight='bold')
scatter2 = ax2.scatter(grouped_R1_I.month, grouped_R1_I.year, c=grouped_R1_I.duration_mean, cmap=cmap, marker='s', vmin=5, vmax=30,s=70)
ax2.text(-0.05,1.01,'('+string.ascii_uppercase[4]+')',fontsize=16, fontweight = 'bold', transform = ax2.transAxes)
scatter3 = ax3.scatter(grouped_R1_I.month, grouped_R1_I.year, c=grouped_R1_I.event_count, cmap=cmap, marker='s', vmin=1, vmax=3,s=70)
ax3.text(-0.05,1.01,'('+string.ascii_uppercase[5]+')',fontsize=16, fontweight = 'bold', transform = ax3.transAxes)
#ax2.set_yticklabels([]);ax3.set_yticklabels([]);
ax1.set_xticklabels([]);ax2.set_xticklabels([]);ax3.set_xticklabels([]);
ax1.set_yticks(years);ax2.set_yticks(years);ax3.set_yticks(years)

# Figure 4, 5, 6
ax4 = fig.add_subplot(gs[6]) # primer heatmsp 
ax5 = fig.add_subplot(gs[7]) # segundo heatmap
ax6 = fig.add_subplot(gs[8]) # segundo heatmap

scatter4 = ax4.scatter(grouped_R4_I.month, grouped_R4_I.year, c=grouped_R4_I.int_mean, cmap=cmap, marker='s', vmin=-1.3, vmax=-0.6,s=70)
ax4.text(-0.05,1.01,'('+string.ascii_uppercase[6]+')',fontsize=16, fontweight = 'bold', transform = ax4.transAxes)
ax4.set_ylabel('Region 4',fontsize=14, fontweight='bold')
scatter5 = ax5.scatter(grouped_R4_I.month, grouped_R4_I.year, c=grouped_R4_I.duration_mean, cmap=cmap, marker='s', vmin=5, vmax=30,s=70)
ax5.text(-0.05,1.01,'('+string.ascii_uppercase[7]+')',fontsize=16, fontweight = 'bold', transform = ax5.transAxes)
scatter6 = ax6.scatter(grouped_R4_I.month, grouped_R4_I.year, c=grouped_R4_I.event_count, cmap=cmap, marker='s', vmin=1, vmax=3,s=70)
ax6.text(-0.05,1.01,'('+string.ascii_uppercase[8]+')',fontsize=16, fontweight = 'bold', transform = ax6.transAxes)

yticks = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
yticklabels = ['Jan', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul','Aug' ,'Sep','Oct','Nov','Dec']

ax4.set_xticks(yticks);ax5.set_xticks(yticks);ax6.set_xticks(yticks)
ax4.set_xticklabels(yticklabels);ax5.set_xticklabels(yticklabels);ax6.set_xticklabels(yticklabels)
ax5.set_yticklabels([]);ax6.set_yticklabels([]);


cax1 = fig.add_subplot(gs[9]) # colorbar 1
cax2 = fig.add_subplot(gs[10]) # colorbar 2
cax3 = fig.add_subplot(gs[11]) # colorbar 2
cbar1 = fig.colorbar(scatter1, cax=cax1,orientation='horizontal')
cbar2 = fig.colorbar(scatter2, cax=cax2,orientation='horizontal')
cbar3 = fig.colorbar(scatter3, cax=cax3,orientation='horizontal')

gs.update(wspace=0.20,hspace=0.15) # Ajustar espaciamiento entre
cbar1.ax.set_xlabel('°C',fontsize=12)
cbar2.ax.set_xlabel('Days',fontsize=12)
cbar3.ax.set_xlabel('Events',fontsize=12)

# Set y-ticks for the second and third rows to be every 5 years
years = np.arange(grouped_R1_I['year'].min(), grouped_R1_I['year'].max() + 1, 5)
ax1.set_yticks(years)
ax2.set_yticks(years)
ax3.set_yticks(years)
ax4.set_yticks(years)
ax5.set_yticks(years)
ax6.set_yticks(years)

fig.savefig('Figura_10.jpeg',dpi=130, bbox_inches='tight')