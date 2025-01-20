import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Predicted Data
## Chronos
B1= pd.read_csv("./raw_data/Box_3_forecast_Chronos_large.csv",sep=",",parse_dates=["run_date","forecast_date"])
B2= pd.read_csv("./raw_data/Box_2_forecast_Chronos_large.csv",sep=",",parse_dates=["run_date","forecast_date"])
B3= pd.read_csv("./raw_data/Box_1_forecast_Chronos_large.csv",sep=",",parse_dates=["run_date","forecast_date"])
B4= pd.read_csv("./raw_data/Box_4_forecast_Chronos_large.csv",sep=",",parse_dates=["run_date","forecast_date"])
## LagLLama
L1= pd.read_csv("./raw_data/Box_3_forecast_LagLlama.csv",sep=",",parse_dates=["run_date","forecast_date"])
L2= pd.read_csv("./raw_data/Box_2_forecast_LagLlama.csv",sep=",",parse_dates=["run_date","forecast_date"])
L3= pd.read_csv("./raw_data/Box_1_forecast_LagLlama.csv",sep=",",parse_dates=["run_date","forecast_date"])
L4= pd.read_csv("./raw_data/Box_4_forecast_LagLlama.csv",sep=",",parse_dates=["run_date","forecast_date"])

# Real Data
B1_R= pd.read_csv("./raw_data/polygon_3_combined.csv",sep=",",parse_dates=["time"])
B1_R["time"] = pd.to_datetime(B1_R["time"]).dt.normalize()
B1_R["analysed_sst"] =B1_R["analysed_sst"]-273.15

B2_R= pd.read_csv("./raw_data/polygon_2_combined.csv",sep=",",parse_dates=["time"])
B2_R["time"] = pd.to_datetime(B2_R["time"]).dt.normalize()
B2_R["analysed_sst"] =B2_R["analysed_sst"]-273.15

B3_R= pd.read_csv("./raw_data/polygon_1_combined.csv",sep=",",parse_dates=["time"])
B3_R["time"] = pd.to_datetime(B3_R["time"]).dt.normalize()
B3_R["analysed_sst"] =B3_R["analysed_sst"]-273.15

B4_R= pd.read_csv("./raw_data/polygon_4_combined.csv",sep=",",parse_dates=["time"])
B4_R["time"] = pd.to_datetime(B4_R["time"]).dt.normalize()
B4_R["analysed_sst"] =B4_R["analysed_sst"]-273.15

# New figure 6 rows and 3 cols
from matplotlib.ticker import FuncFormatter
import matplotlib.dates as mdates
#fig, axes = plt.subplots(6, 3, figsize=(18, 24), sharex=True, sharey=False)
fig, axes = plt.subplots(5, 1, figsize=(15, 14), sharex=True, sharey=False)
######################################################## MAPE ######################################################## 
linew= 0.6; markers= 0.3
# 1 row
axes[0].plot(B1_R.query("time>= '2019-10-04'")['time'], B1_R.query("time>= '2019-10-04'")['analysed_sst'], 
               marker='o', linewidth=linew, label='Region 1', color='blue', markersize=markers)
axes[0].plot(B2_R.query("time>= '2019-10-04'")['time'], B2_R.query("time>= '2019-10-04'")['analysed_sst'], 
               marker='o', linewidth=linew, label='Region 2', color='red', markersize=markers)
axes[0].plot(B3_R.query("time>= '2019-10-04'")['time'], B3_R.query("time>= '2019-10-04'")['analysed_sst'], 
               marker='o', linewidth=linew, label='Region 3', color='violet', markersize=markers)
axes[0].plot(B4_R.query("time>= '2019-10-04'")['time'], B4_R.query("time>= '2019-10-04'")['analysed_sst'], 
               marker='o', linewidth=linew, label='Region 4', color='darkgreen', markersize=markers)
# Adding a vertical span starting from a specific date
#start_date = '2019-10-04'
#axes[0].axvspan(mdates.datestr2num(start_date), xmax=axes[0].get_xlim()[1], color='lightgrey', alpha=0.5, linestyle='--')

axes[0].set_ylabel('SST (°C)', fontsize=14)
axes[0].set_title(f'Sea surface temperature over regions', fontsize=18, fontweight='bold')
axes[0].grid(True, linestyle='--', alpha=0.5)
axes[0].tick_params(axis='both', which='major', labelsize=12)
axes[0].set_ylim([24,32])
axes[0].yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.1f}'))
axes[0].legend(loc='upper left', fontsize=8, ncol=4)
axes[0].text(0.01, 1.18, '(A)', transform=axes[0].transAxes, fontsize=16, fontweight='bold', va='top', ha='right')


# 2 row
Row_1= B1.query("forecast_horizon==1").query("forecast_date>='2019-10-04'").query("forecast_date<='2023-12-31'")['median'].values-\
    B1_R.query("time>= '2019-10-04'")['analysed_sst'].values
Row_2= B1.query("forecast_horizon==7").query("forecast_date>='2019-10-04'").query("forecast_date<='2023-12-31'")['median'].values-\
    B1_R.query("time>= '2019-10-04'")['analysed_sst'].values
Row_3= B1.query("forecast_horizon==14").query("forecast_date>='2019-10-04'").query("forecast_date<='2023-12-31'")['median'].values-\
    B1_R.query("time>= '2019-10-04'")['analysed_sst'].values
Row_4= B1.query("forecast_horizon==28").query("forecast_date>='2019-10-04'").query("forecast_date<='2023-12-31'")['median'].values-\
    B1_R.query("time>= '2019-10-04'")['analysed_sst'].values

axes[1].plot(B1_R.query("time>= '2019-10-04'")['time'], Row_1, marker='o', linewidth=linew, label='D1', color='firebrick', markersize=markers)
axes[1].plot(B1_R.query("time>= '2019-10-04'")['time'], Row_2, marker='o', linewidth=linew, label='D7', color='orange', markersize=markers)
axes[1].plot(B1_R.query("time>= '2019-10-04'")['time'], Row_3, marker='o', linewidth=linew, label='D14', color='skyblue', markersize=markers)
axes[1].plot(B1_R.query("time>= '2019-10-04'")['time'], Row_4, marker='o', linewidth=linew, label='D28', color='gray', markersize=markers)
axes[1].set_ylabel('Bias (°C)', fontsize=14)
axes[1].set_title(f'Region 1', fontsize=18, fontweight='bold')
axes[1].grid(True, linestyle='--', alpha=0.5)
axes[1].tick_params(axis='both', which='major', labelsize=12)
axes[1].set_ylim([-1.5,1.5])
axes[1].text(0.01, 1.18, '(B)', transform=axes[1].transAxes, fontsize=16, fontweight='bold', va='top', ha='right')

# 3 row
Row_1= B2.query("forecast_horizon==1").query("forecast_date>='2019-10-04'").query("forecast_date<='2023-12-31'")['median'].values-\
    B2_R.query("time>= '2019-10-04'")['analysed_sst'].values
Row_2= B2.query("forecast_horizon==7").query("forecast_date>='2019-10-04'").query("forecast_date<='2023-12-31'")['median'].values-\
    B2_R.query("time>= '2019-10-04'")['analysed_sst'].values
Row_3= B2.query("forecast_horizon==14").query("forecast_date>='2019-10-04'").query("forecast_date<='2023-12-31'")['median'].values-\
    B2_R.query("time>= '2019-10-04'")['analysed_sst'].values
Row_4= B2.query("forecast_horizon==28").query("forecast_date>='2019-10-04'").query("forecast_date<='2023-12-31'")['median'].values-\
    B2_R.query("time>= '2019-10-04'")['analysed_sst'].values

axes[2].plot(B2_R.query("time>= '2019-10-04'")['time'], Row_1, marker='o', linewidth=linew, label='D1', color='firebrick', markersize=markers)
axes[2].plot(B2_R.query("time>= '2019-10-04'")['time'], Row_2, marker='o', linewidth=linew, label='D7', color='orange', markersize=markers)
axes[2].plot(B2_R.query("time>= '2019-10-04'")['time'], Row_3, marker='o', linewidth=linew, label='D14', color='skyblue', markersize=markers)
axes[2].plot(B2_R.query("time>= '2019-10-04'")['time'], Row_4, marker='o', linewidth=linew, label='D28', color='gray', markersize=markers)
axes[2].set_ylabel('Bias (°C)', fontsize=14)
axes[2].set_title(f'Region 2', fontsize=18, fontweight='bold')
axes[2].grid(True, linestyle='--', alpha=0.5)
axes[2].tick_params(axis='both', which='major', labelsize=12)
axes[2].set_ylim([-1.5,1.5])
axes[2].text(0.01, 1.18, '(C)', transform=axes[2].transAxes, fontsize=16, fontweight='bold', va='top', ha='right')

# 4 row
Row_1= B3.query("forecast_horizon==1").query("forecast_date>='2019-10-04'").query("forecast_date<='2023-12-31'")['median'].values-\
    B3_R.query("time>= '2019-10-04'")['analysed_sst'].values
Row_2= B3.query("forecast_horizon==7").query("forecast_date>='2019-10-04'").query("forecast_date<='2023-12-31'")['median'].values-\
    B3_R.query("time>= '2019-10-04'")['analysed_sst'].values
Row_3= B3.query("forecast_horizon==14").query("forecast_date>='2019-10-04'").query("forecast_date<='2023-12-31'")['median'].values-\
    B3_R.query("time>= '2019-10-04'")['analysed_sst'].values
Row_4= B3.query("forecast_horizon==28").query("forecast_date>='2019-10-04'").query("forecast_date<='2023-12-31'")['median'].values-\
    B3_R.query("time>= '2019-10-04'")['analysed_sst'].values

axes[3].plot(B3_R.query("time>= '2019-10-04'")['time'], Row_1, marker='o', linewidth=linew, label='D1', color='firebrick', markersize=markers)
axes[3].plot(B3_R.query("time>= '2019-10-04'")['time'], Row_2, marker='o', linewidth=linew, label='D7', color='orange', markersize=markers)
axes[3].plot(B3_R.query("time>= '2019-10-04'")['time'], Row_3, marker='o', linewidth=linew, label='D14', color='skyblue', markersize=markers)
axes[3].plot(B3_R.query("time>= '2019-10-04'")['time'], Row_4, marker='o', linewidth=linew, label='D28', color='gray', markersize=markers)
axes[3].set_ylabel('Bias (°C)', fontsize=14)
axes[3].set_title(f'Region 3', fontsize=18, fontweight='bold')
axes[3].grid(True, linestyle='--', alpha=0.5)
axes[3].tick_params(axis='both', which='major', labelsize=12)
axes[3].set_ylim([-1.5,1.5])
axes[3].text(0.01, 1.18, '(D)', transform=axes[3].transAxes, fontsize=16, fontweight='bold', va='top', ha='right')

# 5 row
Row_1= B4.query("forecast_horizon==1").query("forecast_date>='2019-10-04'").query("forecast_date<='2023-12-31'")['median'].values-\
    B4_R.query("time>= '2019-10-04'")['analysed_sst'].values
Row_2= B4.query("forecast_horizon==7").query("forecast_date>='2019-10-04'").query("forecast_date<='2023-12-31'")['median'].values-\
    B4_R.query("time>= '2019-10-04'")['analysed_sst'].values
Row_3= B4.query("forecast_horizon==14").query("forecast_date>='2019-10-04'").query("forecast_date<='2023-12-31'")['median'].values-\
    B4_R.query("time>= '2019-10-04'")['analysed_sst'].values
Row_4= B4.query("forecast_horizon==28").query("forecast_date>='2019-10-04'").query("forecast_date<='2023-12-31'")['median'].values-\
    B4_R.query("time>= '2019-10-04'")['analysed_sst'].values

axes[4].plot(B4_R.query("time>= '2019-10-04'")['time'], Row_1, marker='o', linewidth=linew, label='D1', color='firebrick', markersize=markers)
axes[4].plot(B4_R.query("time>= '2019-10-04'")['time'], Row_2, marker='o', linewidth=linew, label='D7', color='orange', markersize=markers)
axes[4].plot(B4_R.query("time>= '2019-10-04'")['time'], Row_3, marker='o', linewidth=linew, label='D14', color='skyblue', markersize=markers)
axes[4].plot(B4_R.query("time>= '2019-10-04'")['time'], Row_4, marker='o', linewidth=linew, label='D28', color='gray', markersize=markers)
axes[4].set_ylabel('Bias (°C)', fontsize=14)
axes[4].set_title(f'Region 4', fontsize=18, fontweight='bold')
axes[4].grid(True, linestyle='--', alpha=0.5)
axes[4].tick_params(axis='both', which='major', labelsize=12)
axes[4].set_ylim([-1.5,1.5])
axes[4].text(0.01, 1.18, '(E)', transform=axes[4].transAxes, fontsize=16, fontweight='bold', va='top', ha='right')

handles, labels = axes[4].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=5, fontsize=14, bbox_to_anchor=(0.5, 0.02), bbox_transform=fig.transFigure)

# Adjust the subplot layout to make room for the legend
plt.subplots_adjust(top=0.93, bottom=0.1, hspace=0.3, wspace=0.2)
# Save the figure as a high-resolution JPEG without whitespace
fig.savefig('Figure_6.jpeg', format='jpeg', dpi=300, bbox_inches='tight')