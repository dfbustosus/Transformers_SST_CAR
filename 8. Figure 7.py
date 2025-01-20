import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
import pandas as pd
# Predicted Data
## Chronos
B1= pd.read_csv("./raw_data/Box_3_forecast_Chronos_large.csv",sep=",",parse_dates=["run_date","forecast_date"])
B2= pd.read_csv("./raw_data/Box_2_forecast_Chronos_large.csv",sep=",",parse_dates=["run_date","forecast_date"])
B3= pd.read_csv("./raw_data/Box_1_forecast_Chronos_large.csv",sep=",",parse_dates=["run_date","forecast_date"])
B4= pd.read_csv("./raw_data/Box_4_forecast_Chronos_large.csv",sep=",",parse_dates=["run_date","forecast_date"])

# Real Data
B1_R= pd.read_csv("./raw_data/polygon_3_combined.csv",sep=",",parse_dates=["time"])
B1_R["time"] = pd.to_datetime(B1_R["time"]).dt.normalize()
B2_R= pd.read_csv("./raw_data/polygon_2_combined.csv",sep=",",parse_dates=["time"])
B2_R["time"] = pd.to_datetime(B2_R["time"]).dt.normalize()
B3_R= pd.read_csv("./raw_data/polygon_1_combined.csv",sep=",",parse_dates=["time"])
B3_R["time"] = pd.to_datetime(B3_R["time"]).dt.normalize()
B4_R= pd.read_csv("./raw_data/polygon_4_combined.csv",sep=",",parse_dates=["time"])
B4_R["time"] = pd.to_datetime(B4_R["time"]).dt.normalize()

# Parameters for periodogram, e.g., fs=365 for daily data if we want cycles per year
sampling_frequency = 1  # Set to 1 for cycles per day or adjust to 365 for cycles per year

# Frequencies to highlight (example values in cycles per day for fs=1 or cycles per year for fs=365)
# Expanded list of highlighted frequencies for daily data (fs=365 for cycles per year)
highlight_freqs = {
    #'Annual (1 year)': 1 / 365,
    #'Semi-Annual (6 months)': 2 / 365,
    #'Quarterly (3 months)': 4 / 365,
    #'Bi-Monthly (2 months)': 6 / 365,
    'Monthly': 12 / 365,
    'FortNight': 26 / 365,
    'Weekly': 52 / 365,
    '5 Day': 73 / 365,
    '3 Day': 121.66666667 / 365
}

# Initialize the figure
fig, axes = plt.subplots(4, 1, figsize=(12, 16), sharey=True, sharex=True)
fig.suptitle('Periodograms for MUR-SST, D7, and D28 Forecast Residuals by Region', fontweight='bold')

for i, region in enumerate(mur_sst_data.keys()):
    # Calculate residuals
    d7_residual = mur_sst_data[region] - d7_forecast_data[region]
    d28_residual = mur_sst_data[region] - d28_forecast_data[region]
    
    # Compute periodograms
    freqs_mur, pxx_mur = welch(mur_sst_data[region], fs=sampling_frequency, nperseg=len(mur_sst_data[region])//2)
    freqs_d28, pxx_d28 = welch(d28_residual, fs=sampling_frequency, nperseg=len(d28_residual)//2)
    freqs_d7, pxx_d7 = welch(d7_residual, fs=sampling_frequency, nperseg=len(d7_residual)//2)
    
    # Plot each periodogram in the corresponding panel
    ax = axes[i]
    ax.plot(freqs_mur, pxx_mur, label='MUR-SST', color='blue', alpha=0.5, linewidth=2)
    ax.plot(freqs_d28, pxx_d28, label='D28 Residual', color='green', alpha=0.7, linewidth=1.5)
    ax.plot(freqs_d7, pxx_d7, label='D7 Residual', color='red', alpha=0.9, linewidth=1)
    
    # Highlight specified frequencies with vertical lines
    if i ==0:
        for label, freq in highlight_freqs.items():
            ax.axvline(freq, color='gray', linestyle='--', linewidth=1)
            ax.text(freq, ax.get_ylim()[1]-525, label, rotation=90, verticalalignment='bottom', color='gray')
    else:
        for label, freq in highlight_freqs.items():
            ax.axvline(freq, color='gray', linestyle='--', linewidth=1)
            #ax.text(freq, ax.get_ylim()[1]-200, label, rotation=90, verticalalignment='bottom', color='gray')
    # Label and title adjustments
    ax.set_title(f'{region}', fontweight='bold')
    if i ==3:
        ax.set_xlabel('Frequency (cycles per day)' if sampling_frequency == 1 else 'Frequency (cycles per year)')
    ax.set_ylabel('Spectral Power Density')
    ax.set_yscale('log')  # Optional: use log scale for better clarity in differences
    ax.grid(True, which="both", linestyle='--', linewidth=0.5)

# Add a legend to the first panel only for clarity
axes[0].legend(loc='upper right')

# Adjust layout and display plot
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.subplots_adjust(top=0.93, bottom=0.1, hspace=0.3, wspace=0.2)
# Save the figure as a high-resolution JPEG without whitespace
fig.savefig('Figure_7.jpeg', format='jpeg', dpi=300, bbox_inches='tight')