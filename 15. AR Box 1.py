# 0. Libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error,mean_squared_error, max_error, median_absolute_error
import os
import glob

combined_df = pd.read_csv("./raw_data/polygon_1_combined.csv", parse_dates=["time"])
combined_df["analysed_sst"]=combined_df["analysed_sst"]-273.15
combined_df["time"] = combined_df["time"].dt.strftime('%Y-%m-%d')
combined_df["time"] = pd.to_datetime(combined_df["time"])
print(combined_df.dtypes)

plt.figure(figsize=(18,8))
plt.plot(combined_df.time, combined_df.analysed_sst, linestyle='--',linewidth=0.5)

p = 5  # You can adjust this parameter as needed
from statsmodels.tsa.ar_model import AutoReg

# Create a function to generate forecasts
def generate_forecasts(df, start_date, p):
    forecasts = []
    for run_date in pd.date_range(start=start_date, end=df['time'].max(), freq='D'):
        # Fit the AR model
        model = AutoReg(df['analysed_sst'], lags=p).fit()
        
        # Generate the forecast
        forecast = model.predict(start=len(df), end=len(df) + 29, dynamic=False)
        
        # Create the forecast DataFrame
        forecast_dates = pd.date_range(start=run_date + pd.Timedelta(days=1), periods=30, freq='D')
        forecast_horizon = np.arange(1, 31)
        forecast_df = pd.DataFrame({
            'run_date': run_date,
            'forecast_date': forecast_dates,
            'forecast_horizon': forecast_horizon,
            'median': forecast
        })
        
        forecasts.append(forecast_df)
        
        # Append the forecast to the original data for the next iteration
        # Update the original data with the actual observed values
        actual_data = df[df['time'] <= run_date]
        new_data = pd.DataFrame({'time': forecast_dates, 'analysed_sst': forecast})
        df = pd.concat([actual_data, new_data], ignore_index=True)
            
    return pd.concat(forecasts, ignore_index=True)

# Generate the forecasts starting from 2019-09-06
start_date = '2019-09-06'
forecast_df = generate_forecasts(combined_df, start_date, p)

import pandas as pd
import numpy as np
from statsmodels.tsa.ar_model import AutoReg

def generate_forecasts(df, start_date, p):
    """
    Generate forecasts using a rolling window approach with AutoRegressive model.
    Only uses actual historical data for each prediction window.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe with 'time' and 'analysed_sst' columns
    start_date : str
        Start date for forecasting
    p : int
        Number of lags for the AutoRegressive model
        
    Returns:
    --------
    pandas.DataFrame
        Forecasts with run_date, forecast_date, forecast_horizon, and median columns
    """
    forecasts = []
    
    # Convert start_date to datetime if it's a string
    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date)
    
    # Create date range for forecasting
    date_range = pd.date_range(start=start_date, end=df['time'].max(), freq='D')
    
    # Generate forecasts for each date in the range
    for run_date in date_range:
        # Get historical data up to run_date
        historical_data = df[df['time'] <= run_date]['analysed_sst']
        
        if len(historical_data) <= p:
            continue
            
        try:
            # Fit the AR model on historical data only
            model = AutoReg(historical_data, lags=p).fit()
            
            # Generate the forecast
            forecast = model.predict(start=len(historical_data), 
                                  end=len(historical_data) + 29, 
                                  dynamic=False)
            
            # Create forecast dates
            forecast_dates = pd.date_range(start=run_date + pd.Timedelta(days=1), 
                                         periods=30, freq='D')
            
            # Create forecast DataFrame
            forecast_df = pd.DataFrame({
                'run_date': run_date,
                'forecast_date': forecast_dates,
                'forecast_horizon': np.arange(1, 31),
                'median': forecast
            })
            
            forecasts.append(forecast_df)
            
        except Exception as e:
            print(f"Error generating forecast for {run_date}: {str(e)}")
            continue
    
    # Combine all forecasts
    if not forecasts:
        raise ValueError("No forecasts were generated. Check your data and parameters.")
        
    return pd.concat(forecasts, ignore_index=True)

# Load and prepare the data
def prepare_data(filepath):
    """
    Load and prepare the time series data.
    
    Parameters:
    -----------
    filepath : str
        Path to the CSV file
        
    Returns:
    --------
    pandas.DataFrame
        Prepared dataframe with datetime index
    """
    df = pd.read_csv(filepath, parse_dates=['time'])
    df = df.sort_values(by='time')
    df = df.dropna(subset=['analysed_sst'])
    return df

# Main execution
if __name__ == "__main__":
    # Load and prepare the data
    
    # Set parameters
    p = 5  # AR model lag order
    start_date = '2019-09-06'
    
    try:
        # Generate forecasts
        forecast_df = generate_forecasts(combined_df, start_date, p)
        
        # Save forecasts
        forecast_df.to_csv("./raw_data/Box_1_forecast_AR.csv",
                        sep=",",index=False)
        
        # Print summary
        print("\nForecast Summary:")
        print(f"Total forecasts generated: {len(forecast_df)}")
        print("\nFirst few forecasts:")
        print(forecast_df.head())
        
    except Exception as e:
        print(f"Error in forecast generation: {str(e)}")