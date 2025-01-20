# 0. Libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from chronos import ChronosPipeline
import warnings
warnings.filterwarnings("ignore")
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error,mean_squared_error, max_error, median_absolute_error
import cmocean
import os
import glob

combined_df = pd.read_csv("./raw_data/polygon_1_combined.csv", parse_dates=["time"])
combined_df["analysed_sst"]=combined_df["analysed_sst"]-273.15
combined_df["time"] = combined_df["time"].dt.strftime('%Y-%m-%d')
combined_df["time"] = pd.to_datetime(combined_df["time"])
print(combined_df.dtypes)

plt.figure(figsize=(18,8))
plt.plot(combined_df.time, combined_df.analysed_sst, linestyle='--',linewidth=0.5)

pipeline = ChronosPipeline.from_pretrained(
  "amazon/chronos-t5-large",
  device_map="cuda",
  torch_dtype=torch.bfloat16,
  temperature=0.01,      # already set in your initialization, but can be overridden here
  #top_k=50,              # adjust as needed
  #top_p=0.95,             # adjust as needed
)

print("Training dates: ",
      combined_df[0:6307].time.min(), ' a ',combined_df[0:6307].time.max()
)
print("Testing dates: ",
      combined_df[6307:].time.min(), ' a ',combined_df[6307:].time.max()
)

from tqdm.notebook import tqdm
from IPython.display import display, clear_output
import time
import psutil

def log_message(message):
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}")

def log_memory_usage():
    process = psutil.Process()
    mem_info = process.memory_info()
    log_message(f"Memory usage: {mem_info.rss / 1024 / 1024:.2f} MB")

def generate_forecasts(df, start_date, end_date, pipeline, prediction_length=30):
    results = []
    
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    
    log_message(f"Starting forecast generation from {start_date} to {end_date}")
    
    mask = (df['time'] >= start_date) & (df['time'] <= end_date)
    data_train = df.loc[mask]
    
    log_message(f"Initial training data shape: {data_train.shape}")
    
    context = torch.tensor(data_train['analysed_sst'].values)
    log_message("Starting initial forecast")
    start_time = time.time()
    forecast = pipeline.predict(context, prediction_length=prediction_length)
    log_message(f"Initial forecast completed in {time.time() - start_time:.2f} seconds")
    
    log_message(f"Initial forecast generated with shape: {forecast.shape}")
    
    current_date = end_date
    last_data_date = df['time'].max()
    
    total_days = (last_data_date - current_date).days + 1
    
    for iteration in tqdm(range(total_days), desc="Generating forecasts"):
        if iteration % 10 == 0:
            log_memory_usage()

        run_date = current_date
        forecast_dates = [run_date + timedelta(days=i) for i in range(1, prediction_length+1)]
        
        low, median, high = np.quantile(forecast[0].numpy(), [0.1, 0.5, 0.9], axis=0)
        
        for i, forecast_date in enumerate(forecast_dates):
            results.append({
                'run_date': run_date,
                'forecast_date': forecast_date,
                'forecast_horizon': i + 1,
                'low': low[i],
                'median': median[i],
                'high': high[i]
            })
        
        current_date += timedelta(days=1)
        
        if current_date <= last_data_date:
            new_data = df[df['time'] == current_date]['analysed_sst'].values
            if len(new_data) > 0:
                context = torch.cat([context[0:], torch.tensor([new_data[0]])])
            else:
                log_message(f"No new data available for {current_date}")
            
            start_time = time.time()
            forecast = pipeline.predict(context, prediction_length=prediction_length)
            log_message(f"Forecast for {current_date} completed in {time.time() - start_time:.2f} seconds")
    
    log_message(f"Forecast generation completed. Total forecasts: {len(results)}")
    return pd.DataFrame(results)

# Usage
start_date = '2002-06-01'
end_date = '2019-09-06'

log_message("Starting forecast generation")
forecast_results = generate_forecasts(combined_df, start_date, end_date, pipeline)
log_message("Forecast generation completed")

forecast_results.to_csv("./raw_data/Box_1_forecast_Chronos_large.csv",
                        sep=",",index=False)