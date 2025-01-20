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

# AR
A1= pd.read_csv("./raw_data/Box_3_forecast_AR.csv",sep=",",parse_dates=["run_date","forecast_date"])
A2= pd.read_csv("./raw_data/Box_2_forecast_AR.csv",sep=",",parse_dates=["run_date","forecast_date"])
A3= pd.read_csv("./raw_data/Box_1_forecast_AR.csv",sep=",",parse_dates=["run_date","forecast_date"])
A4= pd.read_csv("./raw_data/Box_4_forecast_AR.csv",sep=",",parse_dates=["run_date","forecast_date"])
## Prophet
P1= pd.read_csv("./raw_data/Box_3_forecast_Prhophet.csv",sep=",",parse_dates=["run_date","forecast_date"])
P2= pd.read_csv("./raw_data/Box_2_forecast_Prhophet.csv",sep=",",parse_dates=["run_date","forecast_date"])
P3= pd.read_csv("./raw_data/Box_1_forecast_Prhophet.csv",sep=",",parse_dates=["run_date","forecast_date"])
P4= pd.read_csv("./raw_data/Box_4_forecast_Prhophet.csv",sep=",",parse_dates=["run_date","forecast_date"])

# Real Data
B1_R= pd.read_csv("./raw_data/polygon_3_combined.csv",sep=",",parse_dates=["time"])
B1_R["time"] = pd.to_datetime(B1_R["time"]).dt.normalize()
B2_R= pd.read_csv("./raw_data/polygon_2_combined.csv",sep=",",parse_dates=["time"])
B2_R["time"] = pd.to_datetime(B2_R["time"]).dt.normalize()
B3_R= pd.read_csv("./raw_data/polygon_1_combined.csv",sep=",",parse_dates=["time"])
B3_R["time"] = pd.to_datetime(B3_R["time"]).dt.normalize()
B4_R= pd.read_csv("./raw_data/polygon_4_combined.csv",sep=",",parse_dates=["time"])
B4_R["time"] = pd.to_datetime(B4_R["time"]).dt.normalize()

## AR
grouped_A1 = A1.\
    query("run_date>='2019-09-06'").\
        query(f"run_date<='2023-12-01'").\
        rename(columns={"median":"p50","forecast_date":"time"}).\
        merge(B1_R[["time","analysed_sst","analysis_error"]], on="time", how="left").\
        assign(analysed_sst=lambda df: df["analysed_sst"] - 273.15).\
        groupby('forecast_horizon')

metrics_A1 = grouped_A1.apply(lambda x: pd.Series({
    'MAPE': np.mean(np.abs((x['analysed_sst'] - x['p50']) / x['analysed_sst'])) * 100,
    'MSE': np.mean((x['analysed_sst'] - x['p50'])**2),
    'RMSE': np.sqrt(np.mean((x['analysed_sst'] - x['p50'])**2)),
    'Bias': np.mean(x['analysed_sst'] - x['p50']),
    'MedianAbsoluteError': np.median(np.abs(x['analysed_sst'] - x['p50'])),
    'MAE': np.mean(np.abs(x['analysed_sst'] - x['p50']))
})).reset_index()

grouped_A2 = A2.\
    query("run_date>='2019-09-06'").\
        query(f"run_date<='2023-12-01'").\
        rename(columns={"median":"p50","forecast_date":"time"}).\
        merge(B2_R[["time","analysed_sst","analysis_error"]], on="time", how="left").\
        assign(analysed_sst=lambda df: df["analysed_sst"] - 273.15).\
        groupby('forecast_horizon')

metrics_A2 = grouped_A2.apply(lambda x: pd.Series({
    'MAPE': np.mean(np.abs((x['analysed_sst'] - x['p50']) / x['analysed_sst'])) * 100,
    'MSE': np.mean((x['analysed_sst'] - x['p50'])**2),
    'RMSE': np.sqrt(np.mean((x['analysed_sst'] - x['p50'])**2)),
    'Bias': np.mean(x['analysed_sst'] - x['p50']),
    'MedianAbsoluteError': np.median(np.abs(x['analysed_sst'] - x['p50'])),
    'MAE': np.mean(np.abs(x['analysed_sst'] - x['p50']))
})).reset_index()


grouped_A3 = A3.\
    query("run_date>='2019-09-06'").\
        query(f"run_date<='2023-12-01'").\
        rename(columns={"median":"p50","forecast_date":"time"}).\
        merge(B3_R[["time","analysed_sst","analysis_error"]], on="time", how="left").\
        assign(analysed_sst=lambda df: df["analysed_sst"] - 273.15).\
        groupby('forecast_horizon')

metrics_A3 = grouped_A3.apply(lambda x: pd.Series({
    'MAPE': np.mean(np.abs((x['analysed_sst'] - x['p50']) / x['analysed_sst'])) * 100,
    'MSE': np.mean((x['analysed_sst'] - x['p50'])**2),
    'RMSE': np.sqrt(np.mean((x['analysed_sst'] - x['p50'])**2)),
    'Bias': np.mean(x['analysed_sst'] - x['p50']),
    'MedianAbsoluteError': np.median(np.abs(x['analysed_sst'] - x['p50'])),
    'MAE': np.mean(np.abs(x['analysed_sst'] - x['p50']))
})).reset_index()

grouped_A4 = A4.\
    query("run_date>='2019-09-06'").\
        query(f"run_date<='2023-12-01'").\
        rename(columns={"median":"p50","forecast_date":"time"}).\
        merge(B4_R[["time","analysed_sst","analysis_error"]], on="time", how="left").\
        assign(analysed_sst=lambda df: df["analysed_sst"] - 273.15).\
        groupby('forecast_horizon')

metrics_A4 = grouped_A4.apply(lambda x: pd.Series({
    'MAPE': np.mean(np.abs((x['analysed_sst'] - x['p50']) / x['analysed_sst'])) * 100,
    'MSE': np.mean((x['analysed_sst'] - x['p50'])**2),
    'RMSE': np.sqrt(np.mean((x['analysed_sst'] - x['p50'])**2)),
    'Bias': np.mean(x['analysed_sst'] - x['p50']),
    'MedianAbsoluteError': np.median(np.abs(x['analysed_sst'] - x['p50'])),
    'MAE': np.mean(np.abs(x['analysed_sst'] - x['p50']))
})).reset_index()

# Phrophet
grouped_P1 = P1.\
    query("run_date>='2019-09-06'").\
        query(f"run_date<='2023-12-01'").\
        rename(columns={"median":"p50","forecast_date":"time"}).\
        merge(B1_R[["time","analysed_sst","analysis_error"]], on="time", how="left").\
        assign(analysed_sst=lambda df: df["analysed_sst"] - 273.15).\
        groupby('forecast_horizon')

metrics_P1 = grouped_P1.apply(lambda x: pd.Series({
    'MAPE': np.mean(np.abs((x['analysed_sst'] - x['p50']) / x['analysed_sst'])) * 100,
    'MSE': np.mean((x['analysed_sst'] - x['p50'])**2),
    'RMSE': np.sqrt(np.mean((x['analysed_sst'] - x['p50'])**2)),
    'Bias': np.mean(x['analysed_sst'] - x['p50']),
    'MedianAbsoluteError': np.median(np.abs(x['analysed_sst'] - x['p50'])),
    'MAE': np.mean(np.abs(x['analysed_sst'] - x['p50']))
})).reset_index()

grouped_P2 = P2.\
    query("run_date>='2019-09-06'").\
        query(f"run_date<='2023-12-01'").\
        rename(columns={"median":"p50","forecast_date":"time"}).\
        merge(B2_R[["time","analysed_sst","analysis_error"]], on="time", how="left").\
        assign(analysed_sst=lambda df: df["analysed_sst"] - 273.15).\
        groupby('forecast_horizon')

metrics_P2 = grouped_P2.apply(lambda x: pd.Series({
    'MAPE': np.mean(np.abs((x['analysed_sst'] - x['p50']) / x['analysed_sst'])) * 100,
    'MSE': np.mean((x['analysed_sst'] - x['p50'])**2),
    'RMSE': np.sqrt(np.mean((x['analysed_sst'] - x['p50'])**2)),
    'Bias': np.mean(x['analysed_sst'] - x['p50']),
    'MedianAbsoluteError': np.median(np.abs(x['analysed_sst'] - x['p50'])),
    'MAE': np.mean(np.abs(x['analysed_sst'] - x['p50']))
})).reset_index()

grouped_P3 = P3.\
    query("run_date>='2019-09-06'").\
        query(f"run_date<='2023-12-01'").\
        rename(columns={"median":"p50","forecast_date":"time"}).\
        merge(B3_R[["time","analysed_sst","analysis_error"]], on="time", how="left").\
        assign(analysed_sst=lambda df: df["analysed_sst"] - 273.15).\
        groupby('forecast_horizon')

metrics_P3 = grouped_P3.apply(lambda x: pd.Series({
    'MAPE': np.mean(np.abs((x['analysed_sst'] - x['p50']) / x['analysed_sst'])) * 100,
    'MSE': np.mean((x['analysed_sst'] - x['p50'])**2),
    'RMSE': np.sqrt(np.mean((x['analysed_sst'] - x['p50'])**2)),
    'Bias': np.mean(x['analysed_sst'] - x['p50']),
    'MedianAbsoluteError': np.median(np.abs(x['analysed_sst'] - x['p50'])),
    'MAE': np.mean(np.abs(x['analysed_sst'] - x['p50']))
})).reset_index()

grouped_P4 = P4.\
    query("run_date>='2019-09-06'").\
        query(f"run_date<='2023-12-01'").\
        rename(columns={"median":"p50","forecast_date":"time"}).\
        merge(B4_R[["time","analysed_sst","analysis_error"]], on="time", how="left").\
        assign(analysed_sst=lambda df: df["analysed_sst"] - 273.15).\
        groupby('forecast_horizon')

metrics_P4 = grouped_P4.apply(lambda x: pd.Series({
    'MAPE': np.mean(np.abs((x['analysed_sst'] - x['p50']) / x['analysed_sst'])) * 100,
    'MSE': np.mean((x['analysed_sst'] - x['p50'])**2),
    'RMSE': np.sqrt(np.mean((x['analysed_sst'] - x['p50'])**2)),
    'Bias': np.mean(x['analysed_sst'] - x['p50']),
    'MedianAbsoluteError': np.median(np.abs(x['analysed_sst'] - x['p50'])),
    'MAE': np.mean(np.abs(x['analysed_sst'] - x['p50']))
})).reset_index()

## Chronos
grouped_B1 = B1.\
    query("run_date>='2019-09-06'").\
        query(f"run_date<='2023-12-01'").\
        rename(columns={"low":"p10", "median":"p50","high":"p90","forecast_date":"time"}).\
        merge(B1_R[["time","analysed_sst","analysis_error"]], on="time", how="left").\
        assign(analysed_sst=lambda df: df["analysed_sst"] - 273.15).\
        groupby('forecast_horizon')

metrics_B1 = grouped_B1.apply(lambda x: pd.Series({
    'MAPE': np.mean(np.abs((x['analysed_sst'] - x['p50']) / x['analysed_sst'])) * 100,
    'MSE': np.mean((x['analysed_sst'] - x['p50'])**2),
    'RMSE': np.sqrt(np.mean((x['analysed_sst'] - x['p50'])**2)),
    'Bias': np.mean(x['analysed_sst'] - x['p50']),
    'MedianAbsoluteError': np.median(np.abs(x['analysed_sst'] - x['p50'])),
    'MAE': np.mean(np.abs(x['analysed_sst'] - x['p50']))
})).reset_index()

grouped_B2 = B2.\
    query("run_date>='2019-09-06'").\
        query(f"run_date<='2023-12-01'").\
        rename(columns={"low":"p10", "median":"p50","high":"p90","forecast_date":"time"}).\
        merge(B2_R[["time","analysed_sst","analysis_error"]], on="time", how="left").\
        assign(analysed_sst=lambda df: df["analysed_sst"] - 273.15).\
        groupby('forecast_horizon')

metrics_B2 = grouped_B2.apply(lambda x: pd.Series({
    'MAPE': np.mean(np.abs((x['analysed_sst'] - x['p50']) / x['analysed_sst'])) * 100,
    'MSE': np.mean((x['analysed_sst'] - x['p50'])**2),
    'RMSE': np.sqrt(np.mean((x['analysed_sst'] - x['p50'])**2)),
    'Bias': np.mean(x['analysed_sst'] - x['p50']),
    'MedianAbsoluteError': np.median(np.abs(x['analysed_sst'] - x['p50'])),
    'MAE': np.mean(np.abs(x['analysed_sst'] - x['p50']))
})).reset_index()

grouped_B3 = B3.\
    query("run_date>='2019-09-06'").\
        query(f"run_date<='2023-12-01'").\
        rename(columns={"low":"p10", "median":"p50","high":"p90","forecast_date":"time"}).\
        merge(B3_R[["time","analysed_sst","analysis_error"]], on="time", how="left").\
        assign(analysed_sst=lambda df: df["analysed_sst"] - 273.15).\
        groupby('forecast_horizon')

metrics_B3 = grouped_B3.apply(lambda x: pd.Series({
    'MAPE': np.mean(np.abs((x['analysed_sst'] - x['p50']) / x['analysed_sst'])) * 100,
    'MSE': np.mean((x['analysed_sst'] - x['p50'])**2),
    'RMSE': np.sqrt(np.mean((x['analysed_sst'] - x['p50'])**2)),
    'Bias': np.mean(x['analysed_sst'] - x['p50']),
    'MedianAbsoluteError': np.median(np.abs(x['analysed_sst'] - x['p50'])),
    'MAE': np.mean(np.abs(x['analysed_sst'] - x['p50']))
})).reset_index()

grouped_B4 = B4.\
    query("run_date>='2019-09-06'").\
        query(f"run_date<='2023-12-01'").\
        rename(columns={"low":"p10", "median":"p50","high":"p90","forecast_date":"time"}).\
        merge(B4_R[["time","analysed_sst","analysis_error"]], on="time", how="left").\
        assign(analysed_sst=lambda df: df["analysed_sst"] - 273.15).\
        groupby('forecast_horizon')

metrics_B4 = grouped_B4.apply(lambda x: pd.Series({
    'MAPE': np.mean(np.abs((x['analysed_sst'] - x['p50']) / x['analysed_sst'])) * 100,
    'MSE': np.mean((x['analysed_sst'] - x['p50'])**2),
    'RMSE': np.sqrt(np.mean((x['analysed_sst'] - x['p50'])**2)),
    'Bias': np.mean(x['analysed_sst'] - x['p50']),
    'MedianAbsoluteError': np.median(np.abs(x['analysed_sst'] - x['p50'])),
    'MAE': np.mean(np.abs(x['analysed_sst'] - x['p50']))
})).reset_index()

###############################################################################################
# Lag LLama
grouped_L1 = L1.\
    query("run_date>='2019-09-06'").\
        query(f"run_date<='2023-12-01'").\
        rename(columns={"forecast":"p50","forecast_date":"time"}).\
        merge(B1_R[["time","analysed_sst","analysis_error"]], on="time", how="left").\
        assign(analysed_sst=lambda df: df["analysed_sst"] - 273.15).\
        groupby('forecast_horizon')

metrics_L1 = grouped_L1.apply(lambda x: pd.Series({
    'MAPE': np.mean(np.abs((x['analysed_sst'] - x['p50']) / x['analysed_sst'])) * 100,
    'MSE': np.mean((x['analysed_sst'] - x['p50'])**2),
    'RMSE': np.sqrt(np.mean((x['analysed_sst'] - x['p50'])**2)),
    'Bias': np.mean(x['analysed_sst'] - x['p50']),
    'MedianAbsoluteError': np.median(np.abs(x['analysed_sst'] - x['p50'])),
    'MAE': np.mean(np.abs(x['analysed_sst'] - x['p50']))
})).reset_index()

grouped_L2 = L2.\
    query("run_date>='2019-09-06'").\
        query(f"run_date<='2023-12-01'").\
        rename(columns={"forecast":"p50","forecast_date":"time"}).\
        merge(B2_R[["time","analysed_sst","analysis_error"]], on="time", how="left").\
        assign(analysed_sst=lambda df: df["analysed_sst"] - 273.15).\
        groupby('forecast_horizon')

metrics_L2 = grouped_L2.apply(lambda x: pd.Series({
    'MAPE': np.mean(np.abs((x['analysed_sst'] - x['p50']) / x['analysed_sst'])) * 100,
    'MSE': np.mean((x['analysed_sst'] - x['p50'])**2),
    'RMSE': np.sqrt(np.mean((x['analysed_sst'] - x['p50'])**2)),
    'Bias': np.mean(x['analysed_sst'] - x['p50']),
    'MedianAbsoluteError': np.median(np.abs(x['analysed_sst'] - x['p50'])),
    'MAE': np.mean(np.abs(x['analysed_sst'] - x['p50']))
})).reset_index()

grouped_L3 = L3.\
    query("run_date>='2019-09-06'").\
        query(f"run_date<='2023-12-01'").\
        rename(columns={"forecast":"p50","forecast_date":"time"}).\
        merge(B3_R[["time","analysed_sst","analysis_error"]], on="time", how="left").\
        assign(analysed_sst=lambda df: df["analysed_sst"] - 273.15).\
        groupby('forecast_horizon')

metrics_L3 = grouped_L3.apply(lambda x: pd.Series({
    'MAPE': np.mean(np.abs((x['analysed_sst'] - x['p50']) / x['analysed_sst'])) * 100,
    'MSE': np.mean((x['analysed_sst'] - x['p50'])**2),
    'RMSE': np.sqrt(np.mean((x['analysed_sst'] - x['p50'])**2)),
    'Bias': np.mean(x['analysed_sst'] - x['p50']),
    'MedianAbsoluteError': np.median(np.abs(x['analysed_sst'] - x['p50'])),
    'MAE': np.mean(np.abs(x['analysed_sst'] - x['p50']))
})).reset_index()

grouped_L4 = L4.\
    query("run_date>='2019-09-06'").\
        query(f"run_date<='2023-12-01'").\
        rename(columns={"forecast":"p50","forecast_date":"time"}).\
        merge(B4_R[["time","analysed_sst","analysis_error"]], on="time", how="left").\
        assign(analysed_sst=lambda df: df["analysed_sst"] - 273.15).\
        groupby('forecast_horizon')

metrics_L4 = grouped_L4.apply(lambda x: pd.Series({
    'MAPE': np.mean(np.abs((x['analysed_sst'] - x['p50']) / x['analysed_sst'])) * 100,
    'MSE': np.mean((x['analysed_sst'] - x['p50'])**2),
    'RMSE': np.sqrt(np.mean((x['analysed_sst'] - x['p50'])**2)),
    'Bias': np.mean(x['analysed_sst'] - x['p50']),
    'MedianAbsoluteError': np.median(np.abs(x['analysed_sst'] - x['p50'])),
    'MAE': np.mean(np.abs(x['analysed_sst'] - x['p50']))
})).reset_index()

### Plot
# New figure 6 rows and 3 cols
from matplotlib.ticker import FuncFormatter
#fig, axes = plt.subplots(6, 3, figsize=(18, 24), sharex=True, sharey=False)
fig, axes = plt.subplots(6, 4, figsize=(15, 10), sharex=True, sharey=False)
######################################################## MAPE ######################################################## 
linew= 0.8; markers= 3
# 1 row, 1 Col MAPE
axes[0,0].plot(metrics_B1['forecast_horizon'], metrics_B1['MAPE'], marker='o', linewidth=linew, label='Chronos', color='blue', markersize=markers)
axes[0,0].plot(metrics_L1['forecast_horizon'], metrics_L1['MAPE'], marker='o', linewidth=linew, label='Lag-LLama', color='red', markersize=markers)
axes[0,0].plot(metrics_A1['forecast_horizon'], metrics_A1['MAPE'], marker='o', linewidth=linew, label='ARIMA', color='green', markersize=markers)
axes[0,0].plot(metrics_P1['forecast_horizon'], metrics_P1['MAPE'], marker='o', linewidth=linew, label='Prophet', color='purple', markersize=markers)
axes[0,0].set_ylabel('MAPE (%)', fontsize=14)
axes[0,0].set_title(f'Region 1', fontsize=18, fontweight='bold')
axes[0,0].grid(True, linestyle='--', alpha=0.7)
axes[0,0].tick_params(axis='both', which='major', labelsize=12)
axes[0,0].set_ylim([0.2,6])
axes[0, 0].yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.1f}'))

# 1 row, 2 Col MAPE
axes[0,1].plot(metrics_B2['forecast_horizon'], metrics_B2['MAPE'], marker='o', linewidth=linew, label='', color='blue', markersize=markers)
axes[0,1].plot(metrics_L2['forecast_horizon'], metrics_L2['MAPE'], marker='o', linewidth=linew, label='Lag-LLama', color='red', markersize=markers)
axes[0,1].plot(metrics_A2['forecast_horizon'], metrics_A2['MAPE'], marker='o', linewidth=linew, label='ARIMA', color='green', markersize=markers)
axes[0,1].plot(metrics_P2['forecast_horizon'], metrics_P2['MAPE'], marker='o', linewidth=linew, label='Prophet', color='purple', markersize=markers)
axes[0,1].set_title(f'Region 2', fontsize=18, fontweight='bold')
axes[0,1].grid(True, linestyle='--', alpha=0.7)
axes[0,1].tick_params(axis='both', which='major', labelsize=12)
axes[0,1].set_ylim([0.2,10])
#axes[0,1].yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.1f}'))
axes[0,1].yaxis.set_major_formatter(plt.NullFormatter())  # Hides y-axis labels

# 1 row, 3 Col MAPE
axes[0,2].plot(metrics_B3['forecast_horizon'], metrics_B3['MAPE'], marker='o', linewidth=linew, label='', color='blue', markersize=markers)
axes[0,2].plot(metrics_L3['forecast_horizon'], metrics_L3['MAPE'], marker='o', linewidth=linew, label='Lag-LLama', color='red', markersize=markers)
axes[0,2].plot(metrics_A3['forecast_horizon'], metrics_A3['MAPE'], marker='o', linewidth=linew, label='ARIMA', color='green', markersize=markers)
axes[0,2].plot(metrics_P3['forecast_horizon'], metrics_P3['MAPE'], marker='o', linewidth=linew, label='Prophet', color='purple', markersize=markers)
axes[0,2].set_title(f'Region 3', fontsize=18, fontweight='bold')
axes[0,2].grid(True, linestyle='--', alpha=0.7)
axes[0,2].tick_params(axis='both', which='major', labelsize=12)
axes[0,2].set_ylim([0.2,6])
#axes[0,2].yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.1f}'))
axes[0,2].yaxis.set_major_formatter(plt.NullFormatter())  # Hides y-axis labels

# 1 row, 4 Col MAPE
axes[0,3].plot(metrics_B4['forecast_horizon'], metrics_B4['MAPE'], marker='o', linewidth=linew, label='', color='blue', markersize=markers)
axes[0,3].plot(metrics_L4['forecast_horizon'], metrics_L4['MAPE'], marker='o', linewidth=linew, label='Lag-LLama', color='red', markersize=markers)
axes[0,3].plot(metrics_A4['forecast_horizon'], metrics_A4['MAPE'], marker='o', linewidth=linew, label='ARIMA', color='green', markersize=markers)
axes[0,3].plot(metrics_P4['forecast_horizon'], metrics_P4['MAPE'], marker='o', linewidth=linew, label='Prophet' , color='purple', markersize=markers)
axes[0,3].set_title(f'Region 4', fontsize=18, fontweight='bold')
axes[0,3].grid(True, linestyle='--', alpha=0.7)
axes[0,3].tick_params(axis='both', which='major', labelsize=12)
axes[0,3].set_ylim([0.2,6])
#axes[0,3].yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.1f}'))
axes[0,3].yaxis.set_major_formatter(plt.NullFormatter())  # Hides y-axis labels

######################################################## MSE ######################################################## 
# 2 row, 1 Col MSE
axes[1,0].plot(metrics_B1['forecast_horizon'], metrics_B1['MSE'], marker='o', linewidth=linew, label='', color='blue', markersize=markers)
axes[1,0].plot(metrics_L1['forecast_horizon'], metrics_L1['MSE'], marker='o', linewidth=linew, label='', color='red', markersize=markers)
axes[1,0].plot(metrics_A1['forecast_horizon'], metrics_A1['MSE'], marker='o', linewidth=linew, label='', color='green', markersize=markers)
axes[1,0].plot(metrics_P1['forecast_horizon'], metrics_P1['MSE'], marker='o', linewidth=linew, label='', color='purple', markersize=markers)
axes[1,0].set_ylabel('MSE ($°C^2$)', fontsize=14)
axes[1,0].grid(True, linestyle='--', alpha=0.7)
axes[1,0].tick_params(axis='both', which='major', labelsize=12)
axes[1,0].set_ylim([0,5])
axes[1,0].yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.1f}'))

# 2 row, 2 Col MSE
axes[1,1].plot(metrics_B2['forecast_horizon'], metrics_B2['MSE'], marker='o', linewidth=linew, label='', color='blue', markersize=markers)
axes[1,1].plot(metrics_L2['forecast_horizon'], metrics_L2['MSE'], marker='o', linewidth=linew, label='', color='red', markersize=markers)
axes[1,1].plot(metrics_A2['forecast_horizon'], metrics_A2['MSE'], marker='o', linewidth=linew, label='', color='green', markersize=markers)
axes[1,1].plot(metrics_P2['forecast_horizon'], metrics_P2['MSE'], marker='o', linewidth=linew, label='', color='purple', markersize=markers)
axes[1,1].grid(True, linestyle='--', alpha=0.7)
axes[1,1].tick_params(axis='both', which='major', labelsize=12)
axes[1,1].set_ylim([0,5])
#axes[1,1].yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.1f}'))
axes[1,1].yaxis.set_major_formatter(plt.NullFormatter())  # Hides y-axis labels

# 2 row, 3 Col MSE
axes[1,2].plot(metrics_B3['forecast_horizon'], metrics_B3['MSE'], marker='o', linewidth=linew, label='', color='blue', markersize=markers)
axes[1,2].plot(metrics_L3['forecast_horizon'], metrics_L3['MSE'], marker='o', linewidth=linew, label='', color='red', markersize=markers)
axes[1,2].plot(metrics_A3['forecast_horizon'], metrics_A3['MSE'], marker='o', linewidth=linew, label='', color='green', markersize=markers)
axes[1,2].plot(metrics_P3['forecast_horizon'], metrics_P3['MSE'], marker='o', linewidth=linew, label='', color='purple', markersize=markers)
axes[1,2].grid(True, linestyle='--', alpha=0.7)
axes[1,2].tick_params(axis='both', which='major', labelsize=12)
axes[1,2].set_ylim([0,5])
#axes[1,2].yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.1f}'))
axes[1,2].yaxis.set_major_formatter(plt.NullFormatter())  # Hides y-axis labels


# 2 row, 4 Col MSE
axes[1,3].plot(metrics_B4['forecast_horizon'], metrics_B4['MSE'], marker='o', linewidth=linew, label='', color='blue', markersize=markers)
axes[1,3].plot(metrics_L4['forecast_horizon'], metrics_L4['MSE'], marker='o', linewidth=linew, label='', color='red', markersize=markers)
axes[1,3].plot(metrics_A4['forecast_horizon'], metrics_A4['MSE'], marker='o', linewidth=linew, label='', color='green', markersize=markers)
axes[1,3].plot(metrics_P4['forecast_horizon'], metrics_P4['MSE'], marker='o', linewidth=linew, label='', color='purple', markersize=markers)
axes[1,3].grid(True, linestyle='--', alpha=0.7)
axes[1,3].tick_params(axis='both', which='major', labelsize=12)
axes[1,3].set_ylim([0,5])
#axes[1,3].yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.1f}'))
axes[1,3].yaxis.set_major_formatter(plt.NullFormatter())  # Hides y-axis labels


######################################################## RMSE ######################################################## 
# 3 row, 1 Col RMSE
axes[2,0].plot(metrics_B1['forecast_horizon'], metrics_B1['RMSE'], marker='o', linewidth=linew, label='', color='blue', markersize=markers)
axes[2,0].plot(metrics_L1['forecast_horizon'], metrics_L1['RMSE'], marker='o', linewidth=linew, label='', color='red', markersize=markers)
axes[2,0].plot(metrics_A1['forecast_horizon'], metrics_A1['RMSE'], marker='o', linewidth=linew, label='', color='green', markersize=markers)
axes[2,0].plot(metrics_P1['forecast_horizon'], metrics_P1['RMSE'], marker='o', linewidth=linew, label='', color='purple', markersize=markers)
axes[2,0].set_ylabel('RMSE ($°C$)', fontsize=14)
axes[2,0].grid(True, linestyle='--', alpha=0.7)
axes[2,0].tick_params(axis='both', which='major', labelsize=12)
axes[2,0].set_ylim([0.1,2.2])
axes[2,0].yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.1f}'))


# 3 row, 2 Col RMSE
axes[2,1].plot(metrics_B2['forecast_horizon'], metrics_B2['RMSE'], marker='o', linewidth=linew, label='', color='blue', markersize=markers)
axes[2,1].plot(metrics_L2['forecast_horizon'], metrics_L2['RMSE'], marker='o', linewidth=linew, label='', color='red', markersize=markers)
axes[2,1].plot(metrics_A2['forecast_horizon'], metrics_A2['RMSE'], marker='o', linewidth=linew, label='', color='green', markersize=markers)
axes[2,1].plot(metrics_P2['forecast_horizon'], metrics_P2['RMSE'], marker='o', linewidth=linew, label='', color='purple', markersize=markers)
axes[2,1].grid(True, linestyle='--', alpha=0.7)
axes[2,1].tick_params(axis='both', which='major', labelsize=12)
axes[2,1].set_ylim([0.1,2.2])
#axes[2,1].yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.1f}'))
axes[2,1].yaxis.set_major_formatter(plt.NullFormatter())  # Hides y-axis labels

# 3 row, 3 Col RMSE
axes[2,2].plot(metrics_B3['forecast_horizon'], metrics_B3['RMSE'], marker='o', linewidth=linew, label='', color='blue', markersize=markers)
axes[2,2].plot(metrics_L3['forecast_horizon'], metrics_L3['RMSE'], marker='o', linewidth=linew, label='', color='red', markersize=markers)
axes[2,2].plot(metrics_A3['forecast_horizon'], metrics_A3['RMSE'], marker='o', linewidth=linew, label='', color='green', markersize=markers)
axes[2,2].plot(metrics_P3['forecast_horizon'], metrics_P3['RMSE'], marker='o', linewidth=linew, label='', color='purple', markersize=markers)
axes[2,2].grid(True, linestyle='--', alpha=0.7)
axes[2,2].tick_params(axis='both', which='major', labelsize=12)
axes[2,2].set_ylim([0.1,2.2])
#axes[2,2].yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.1f}'))
axes[2,2].yaxis.set_major_formatter(plt.NullFormatter())  # Hides y-axis labels

# 3 row, 4 Col RMSE
axes[2,3].plot(metrics_B4['forecast_horizon'], metrics_B4['RMSE'], marker='o', linewidth=linew, label='', color='blue', markersize=markers)
axes[2,3].plot(metrics_L4['forecast_horizon'], metrics_L4['RMSE'], marker='o', linewidth=linew, label='', color='red', markersize=markers)
axes[2,3].plot(metrics_A4['forecast_horizon'], metrics_A4['RMSE'], marker='o', linewidth=linew, label='', color='green', markersize=markers)
axes[2,3].plot(metrics_P4['forecast_horizon'], metrics_P4['RMSE'], marker='o', linewidth=linew, label='', color='purple', markersize=markers)
axes[2,3].grid(True, linestyle='--', alpha=0.7)
axes[2,3].tick_params(axis='both', which='major', labelsize=12)
axes[2,3].set_ylim([0.1,2.2])
#axes[2,3].yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.1f}'))
axes[2,3].yaxis.set_major_formatter(plt.NullFormatter())  # Hides y-axis labels


######################################################## Bias ######################################################## 
# 4 row, 1 Col Bias
axes[3,0].plot(metrics_B1['forecast_horizon'], metrics_B1['Bias'], marker='o', linewidth=linew, label='', color='blue', markersize=markers)
axes[3,0].plot(metrics_L1['forecast_horizon'], metrics_L1['Bias'], marker='o', linewidth=linew, label='', color='red', markersize=markers)
axes[3,0].plot(metrics_A1['forecast_horizon'], metrics_A1['Bias'], marker='o', linewidth=linew, label='', color='green', markersize=markers)
axes[3,0].plot(metrics_P1['forecast_horizon'], metrics_P1['Bias'], marker='o', linewidth=linew, label='', color='purple', markersize=markers)
axes[3,0].set_ylabel('BIAS ($°C$)', fontsize=14)
axes[3,0].grid(True, linestyle='--', alpha=0.7)
axes[3,0].tick_params(axis='both', which='major', labelsize=12)
axes[3,0].set_ylim([-0.2,0.35])
axes[3,0].yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.1f}'))

# 4 row, 2 Col Bias
axes[3,1].plot(metrics_B2['forecast_horizon'], metrics_B2['Bias'], marker='o', linewidth=linew, label='', color='blue', markersize=markers)
axes[3,1].plot(metrics_L2['forecast_horizon'], metrics_L2['Bias'], marker='o', linewidth=linew, label='', color='red', markersize=markers)
axes[3,1].plot(metrics_A2['forecast_horizon'], metrics_A2['Bias'], marker='o', linewidth=linew, label='', color='green', markersize=markers)
axes[3,1].plot(metrics_P2['forecast_horizon'], metrics_P2['Bias'], marker='o', linewidth=linew, label='', color='purple', markersize=markers)
axes[3,1].grid(True, linestyle='--', alpha=0.7)
axes[3,1].tick_params(axis='both', which='major', labelsize=12)
axes[3,1].set_ylim([-0.2,0.35])
#axes[3,1].yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.1f}'))
axes[3,1].yaxis.set_major_formatter(plt.NullFormatter())  # Hides y-axis labels

# 4 row, 3 Col Bias
axes[3,2].plot(metrics_B3['forecast_horizon'], metrics_B3['Bias'], marker='o', linewidth=linew, label='', color='blue', markersize=markers)
axes[3,2].plot(metrics_L3['forecast_horizon'], metrics_L3['Bias'], marker='o', linewidth=linew, label='', color='red', markersize=markers)
axes[3,2].plot(metrics_A3['forecast_horizon'], metrics_A3['Bias'], marker='o', linewidth=linew, label='', color='green', markersize=markers)
axes[3,2].plot(metrics_P3['forecast_horizon'], metrics_P3['Bias'], marker='o', linewidth=linew, label='', color='purple', markersize=markers)
axes[3,2].grid(True, linestyle='--', alpha=0.7)
axes[3,2].tick_params(axis='both', which='major', labelsize=12)
axes[3,2].set_ylim([-0.2,0.35])
#axes[3,2].yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.1f}'))
axes[3,2].yaxis.set_major_formatter(plt.NullFormatter())  # Hides y-axis labels

# 4 row, 4 Col Bias
axes[3,3].plot(metrics_B4['forecast_horizon'], metrics_B4['Bias'], marker='o', linewidth=linew, label='', color='blue', markersize=markers)
axes[3,3].plot(metrics_L4['forecast_horizon'], metrics_L4['Bias'], marker='o', linewidth=linew, label='', color='red', markersize=markers)
axes[3,3].plot(metrics_A4['forecast_horizon'], metrics_A3['Bias'], marker='o', linewidth=linew, label='', color='green', markersize=markers)
axes[3,3].plot(metrics_P4['forecast_horizon'], metrics_P3['Bias'], marker='o', linewidth=linew, label='', color='purple', markersize=markers)
axes[3,3].grid(True, linestyle='--', alpha=0.7)
axes[3,3].tick_params(axis='both', which='major', labelsize=12)
axes[3,3].set_ylim([-0.2,0.35])
#axes[3,3].yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.1f}'))
axes[3,3].yaxis.set_major_formatter(plt.NullFormatter())  # Hides y-axis labels

######################################################## MedAbsError ######################################################## 
# 5 row, 1 Col MedianAbsoluteError
axes[4,0].plot(metrics_B1['forecast_horizon'], metrics_B1['MedianAbsoluteError'], marker='o', linewidth=linew, label='', color='blue', markersize=markers)
axes[4,0].plot(metrics_L1['forecast_horizon'], metrics_L1['MedianAbsoluteError'], marker='o', linewidth=linew, label='', color='red', markersize=markers)
axes[4,0].plot(metrics_A1['forecast_horizon'], metrics_A1['MedianAbsoluteError'], marker='o', linewidth=linew, label='', color='green', markersize=markers)
axes[4,0].plot(metrics_P1['forecast_horizon'], metrics_P1['MedianAbsoluteError'], marker='o', linewidth=linew, label='', color='purple', markersize=markers)
axes[4,0].set_ylabel('MedABS ($°C$)', fontsize=14)
axes[4,0].grid(True, linestyle='--', alpha=0.7)
axes[4,0].tick_params(axis='both', which='major', labelsize=12)
axes[4,0].set_ylim([0.05,1.5])
axes[4,0].yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.1f}'))

# 5 row, 2 Col MedianAbsoluteError
axes[4,1].plot(metrics_B2['forecast_horizon'], metrics_B2['MedianAbsoluteError'], marker='o', linewidth=linew, label='', color='blue', markersize=markers)
axes[4,1].plot(metrics_L2['forecast_horizon'], metrics_L2['MedianAbsoluteError'], marker='o', linewidth=linew, label='', color='red', markersize=markers)
axes[4,1].plot(metrics_A2['forecast_horizon'], metrics_A2['MedianAbsoluteError'], marker='o', linewidth=linew, label='', color='green', markersize=markers)
axes[4,1].plot(metrics_P2['forecast_horizon'], metrics_P2['MedianAbsoluteError'], marker='o', linewidth=linew, label='', color='purple', markersize=markers)
axes[4,1].grid(True, linestyle='--', alpha=0.7)
axes[4,1].tick_params(axis='both', which='major', labelsize=12)
axes[4,1].set_ylim([0.05,1.5])
#axes[4,1].yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.1f}'))
axes[4,1].yaxis.set_major_formatter(plt.NullFormatter())  # Hides y-axis labels

# 5 row, 3 Col MedianAbsoluteError
axes[4,2].plot(metrics_B3['forecast_horizon'], metrics_B3['MedianAbsoluteError'], marker='o', linewidth=linew, label='', color='blue', markersize=markers)
axes[4,2].plot(metrics_L3['forecast_horizon'], metrics_L3['MedianAbsoluteError'], marker='o', linewidth=linew, label='', color='red', markersize=markers)
axes[4,2].plot(metrics_A3['forecast_horizon'], metrics_A3['MedianAbsoluteError'], marker='o', linewidth=linew, label='', color='green', markersize=markers)
axes[4,2].plot(metrics_P3['forecast_horizon'], metrics_P3['MedianAbsoluteError'], marker='o', linewidth=linew, label='', color='purple', markersize=markers)
axes[4,2].grid(True, linestyle='--', alpha=0.7)
axes[4,2].tick_params(axis='both', which='major', labelsize=12)
axes[4,2].set_ylim([0.05,1.5])
#axes[4,2].yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.1f}'))
axes[4,2].yaxis.set_major_formatter(plt.NullFormatter())  # Hides y-axis labels

# 5 row, 3 Col MedianAbsoluteError
axes[4,3].plot(metrics_B4['forecast_horizon'], metrics_B4['MedianAbsoluteError'], marker='o', linewidth=linew, label='', color='blue', markersize=markers)
axes[4,3].plot(metrics_L4['forecast_horizon'], metrics_L4['MedianAbsoluteError'], marker='o', linewidth=linew, label='', color='red', markersize=markers)
axes[4,3].plot(metrics_A4['forecast_horizon'], metrics_A4['MedianAbsoluteError'], marker='o', linewidth=linew, label='', color='green', markersize=markers)
axes[4,3].plot(metrics_P4['forecast_horizon'], metrics_P4['MedianAbsoluteError'], marker='o', linewidth=linew, label='', color='purple', markersize=markers)
axes[4,3].grid(True, linestyle='--', alpha=0.7)
axes[4,3].tick_params(axis='both', which='major', labelsize=12)
axes[4,3].set_ylim([0.05,1.5])
#axes[4,3].yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.1f}'))
axes[4,3].yaxis.set_major_formatter(plt.NullFormatter())  # Hides y-axis labels

######################################################## MAE ######################################################## 
# 6 row, 1 Col MAE
axes[5,0].plot(metrics_B1['forecast_horizon'], metrics_B1['MAE'], marker='o', linewidth=linew, label='', color='blue', markersize=markers)
axes[5,0].plot(metrics_L1['forecast_horizon'], metrics_L1['MAE'], marker='o', linewidth=linew, label='', color='red', markersize=markers)
axes[5,0].plot(metrics_A1['forecast_horizon'], metrics_A1['MAE'], marker='o', linewidth=linew, label='', color='green', markersize=markers)
axes[5,0].plot(metrics_P1['forecast_horizon'], metrics_P1['MAE'], marker='o', linewidth=linew, label='', color='purple', markersize=markers)
axes[5,0].set_ylabel('MAE ($°C$)', fontsize=14)
axes[5,0].grid(True, linestyle='--', alpha=0.7)
axes[5,0].tick_params(axis='both', which='major', labelsize=12)
axes[5,0].set_ylim([0.09,2])
axes[5,0].set_xlabel('Forecast Horizon (D)')
axes[5,0].yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.1f}'))

# 6 row, 2 Col MAE
axes[5,1].plot(metrics_B2['forecast_horizon'], metrics_B2['MAE'], marker='o', linewidth=linew, label='', color='blue', markersize=markers)
axes[5,1].plot(metrics_L2['forecast_horizon'], metrics_L2['MAE'], marker='o', linewidth=linew, label='', color='red', markersize=markers)
axes[5,1].plot(metrics_A2['forecast_horizon'], metrics_A2['MAE'], marker='o', linewidth=linew, label='', color='green', markersize=markers)
axes[5,1].plot(metrics_P2['forecast_horizon'], metrics_P2['MAE'], marker='o', linewidth=linew, label='', color='purple', markersize=markers)
axes[5,1].grid(True, linestyle='--', alpha=0.7)
axes[5,1].tick_params(axis='both', which='major', labelsize=12)
axes[5,1].set_ylim([0.09,2])
axes[5,1].set_xlabel('Forecast Horizon (D)')
#axes[5,1].yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.1f}'))
axes[5,1].yaxis.set_major_formatter(plt.NullFormatter())  # Hides y-axis labels

# 6 row, 3 Col MAE
axes[5,2].plot(metrics_B3['forecast_horizon'], metrics_B3['MAE'], marker='o', linewidth=linew, label='', color='blue', markersize=markers)
axes[5,2].plot(metrics_L3['forecast_horizon'], metrics_L3['MAE'], marker='o', linewidth=linew, label='', color='red', markersize=markers)
axes[5,2].plot(metrics_A3['forecast_horizon'], metrics_A3['MAE'], marker='o', linewidth=linew, label='', color='green', markersize=markers)
axes[5,2].plot(metrics_P3['forecast_horizon'], metrics_P3['MAE'], marker='o', linewidth=linew, label='', color='purple', markersize=markers)
axes[5,2].grid(True, linestyle='--', alpha=0.7)
axes[5,2].tick_params(axis='both', which='major', labelsize=12)
axes[5,2].set_ylim([0.09,2])
axes[5,2].set_xlabel('Forecast Horizon (D)')
#axes[5,2].yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.1f}'))
axes[5,2].yaxis.set_major_formatter(plt.NullFormatter())  # Hides y-axis labels

# 6 row, 4 Col MAE
axes[5,3].plot(metrics_B4['forecast_horizon'], metrics_B4['MAE'], marker='o', linewidth=linew, label='', color='blue', markersize=markers)
axes[5,3].plot(metrics_L4['forecast_horizon'], metrics_L4['MAE'], marker='o', linewidth=linew, label='', color='red', markersize=markers)
axes[5,3].plot(metrics_A4['forecast_horizon'], metrics_A4['MAE'], marker='o', linewidth=linew, label='', color='green', markersize=markers)
axes[5,3].plot(metrics_P4['forecast_horizon'], metrics_P4['MAE'], marker='o', linewidth=linew, label='', color='purple', markersize=markers)
axes[5,3].grid(True, linestyle='--', alpha=0.7)
axes[5,3].tick_params(axis='both', which='major', labelsize=12)
axes[5,3].set_ylim([0.09,2])
axes[5,3].set_xlabel('Forecast Horizon (D)')
#axes[5,3].yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.1f}'))
axes[5,3].yaxis.set_major_formatter(plt.NullFormatter())  # Hides y-axis labels

handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=5, fontsize=14, bbox_to_anchor=(0.5, -0.01), bbox_transform=fig.transFigure)

# Adjust the subplot layout to make room for the legend
plt.subplots_adjust(top=0.93, bottom=0.1, hspace=0.3, wspace=0.2)

# Save the figure as a high-resolution JPEG without whitespace
fig.savefig('Figure_5.jpeg', format='jpeg', dpi=300, bbox_inches='tight')