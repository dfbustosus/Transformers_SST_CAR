import pandas as pd
import xarray as xr
import os
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
path_SST= "D:/ARCHIVOS E/Data Probability EBUS/MUR/Temporal/"
output_SST= 'D:/ARCHIVOS E/Data Probability EBUS/MUR/Output/'
path_SST_A= "E:/ARCHIVOS E/Data Probability EBUS/MUR/"
raw_path_export= "./raw_data"

current_path = path_SST
# 2010
files = sorted(
    [x for x in os.listdir(current_path) 
     if 'GHRSST-SSTfnd-MUR-GLOB-v02.0-fv04.1' in x and x.startswith('2010')
    ]
)
# Poligno 1
SST_1 = xr.open_dataset(os.path.join(current_path, files[0])).\
    sel(lat=slice(9.5,11.5), lon=slice(-80,-78))
SST_1

# Poligno 2
SST_2 = xr.open_dataset(os.path.join(current_path, files[0])).\
    sel(lat=slice(20,22), lon=slice(-83,-81))
# Poligno 3
SST_3 = xr.open_dataset(os.path.join(current_path, files[0])).\
    sel(lat=slice(10.5,12.5), lon=slice(-68,-66))

# Poligno 4
SST_4 = xr.open_dataset(os.path.join(current_path, files[0])).\
    sel(lat=slice(11.75,13.75), lon=slice(-73.25,-71.25))

# Plots
SST_1.isel(time=0).analysed_sst.plot(cmap='jet')
plt.show()
SST_2.isel(time=0).analysed_sst.plot(cmap='jet')
plt.show()
SST_3.isel(time=0).analysed_sst.plot(cmap='jet')
plt.show()
SST_4.isel(time=0).analysed_sst.plot(cmap='jet')
plt.show()
#del SST_1, SST_2,SST_3


current_path = path_SST
start_year = 2010  # Start year
end_year = 2023    # End year (inclusive)

for year in range(start_year, end_year + 1):
    print(f"Processing year: {year}")
    polygon_1 = []
    polygon_2 = []
    polygon_3 = []
    polygon_4 = []

    # Filter files for the current year
    files = sorted(
        [x for x in os.listdir(current_path)
         if 'GHRSST-SSTfnd-MUR-GLOB-v02.0-fv04.1' in x and x.startswith(str(year))]
    )

    # Loop to get data for the current year
    for i in files:
        print(i)
        SST_1 = xr.open_dataset(os.path.join(current_path, i)).\
            sel(lat=slice(9.5, 11.5), lon=slice(-80, -78)).\
            drop(["mask", "sea_ice_fraction"])
        polygon_1.append(SST_1)
        
        SST_2 = xr.open_dataset(os.path.join(current_path, i)).\
            sel(lat=slice(20, 22), lon=slice(-83, -81)).\
            drop(["mask", "sea_ice_fraction"])
        polygon_2.append(SST_2)
        
        SST_3 = xr.open_dataset(os.path.join(current_path, i)).\
            sel(lat=slice(10.5, 12.5), lon=slice(-68, -66)).\
            drop(["mask", "sea_ice_fraction"])
        polygon_3.append(SST_3)
        
        SST_4 = xr.open_dataset(os.path.join(current_path, i)).\
            sel(lat=slice(11.75, 13.75), lon=slice(-73.25, -71.25)).\
            drop(["mask", "sea_ice_fraction"])
        polygon_4.append(SST_4)

    # Concatenate data in time
    P1 = xr.concat(polygon_1, dim='time')
    P2 = xr.concat(polygon_2, dim='time')
    P3 = xr.concat(polygon_3, dim='time')
    P4 = xr.concat(polygon_4, dim='time')

    # Convert to DataFrame
    D1 = P1.mean(dim=['lat', 'lon']).to_dataframe().reset_index()
    D2 = P2.mean(dim=['lat', 'lon']).to_dataframe().reset_index()
    D3 = P3.mean(dim=['lat', 'lon']).to_dataframe().reset_index()
    D4 = P4.mean(dim=['lat', 'lon']).to_dataframe().reset_index()

    # Export to CSV
    D1.to_csv(f"{raw_path_export}/polygon_1/data_{year}.csv", index=False)
    D2.to_csv(f"{raw_path_export}/polygon_2/data_{year}.csv", index=False)
    D3.to_csv(f"{raw_path_export}/polygon_3/data_{year}.csv", index=False)
    D4.to_csv(f"{raw_path_export}/polygon_4/data_{year}.csv", index=False)

    print(f"Finished processing year: {year}")



def concatenate_csv_files(folder_path, columns_of_interest=None, output_filename='combined_data.csv'):
    """
    Concatenate multiple CSV files in a folder, keeping only specified columns.
    
    Parameters:
    - folder_path (str): Path to the folder containing the CSV files.
    - columns_of_interest (list, optional): List of columns to keep in the final DataFrame.
    - output_filename (str): Name of the output file to save the concatenated data.
    
    Returns:
    - pd.DataFrame: The concatenated DataFrame.
    """
    # List all CSV files in the folder
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    dataframes = []

    # Loop over each file and read the data
    for file in csv_files:
        file_path = os.path.join(folder_path, file)
        # Read the CSV, selecting only the columns of interest if specified
        if columns_of_interest:
            df = pd.read_csv(file_path, usecols=columns_of_interest, parse_dates=["time"])
        else:
            df = pd.read_csv(file_path)
        dataframes.append(df)

    # Concatenate all DataFrames into a single DataFrame
    combined_df = pd.concat(dataframes, ignore_index=True)
    # Export the combined DataFrame to a new CSV file
    #combined_df.to_csv(os.path.join(folder_path, output_filename), index=False)
    #print(f"Concatenation complete. Combined file saved as '{output_filename}'.")
    return combined_df

# Polygon 1
folder_path_1 = raw_path_export + "/polygon_1/"
columns_of_interest = ['time', 'analysed_sst', 'analysis_error']
combined_df_1 = concatenate_csv_files(folder_path_1, columns_of_interest)
# Polygon 2
folder_path_2 = raw_path_export + "/polygon_2/"
combined_df_2 = concatenate_csv_files(folder_path_2, columns_of_interest)
# Polygon 3
folder_path_3 = raw_path_export + "/polygon_3/"
combined_df_3 = concatenate_csv_files(folder_path_3, columns_of_interest)
# Polygon 4
folder_path_4 = raw_path_export + "/polygon_4/"
combined_df_4 = concatenate_csv_files(folder_path_4, columns_of_interest)
combined_df_4

# Export data
combined_df_1.to_csv(os.path.join(raw_path_export, "polygon_1_combined.csv"), index=False)
combined_df_2.to_csv(os.path.join(raw_path_export, "polygon_2_combined.csv"), index=False)
combined_df_3.to_csv(os.path.join(raw_path_export, "polygon_3_combined.csv"), index=False)
combined_df_4.to_csv(os.path.join(raw_path_export, "polygon_4_combined.csv"), index=False)

combined_df_1.set_index("time")["analysed_sst"].plot(figsize=(19,6))
combined_df_2.set_index("time")["analysed_sst"].plot(figsize=(19,6))
combined_df_3.set_index("time")["analysed_sst"].plot(figsize=(19,6))
combined_df_4.set_index("time")["analysed_sst"].plot(figsize=(19,6))


path_SST = "D:/ARCHIVOS E/Data Probability EBUS/MUR/Temporal/"
year = '2023'
current_path = path_SST

files = sorted(
    [x for x in os.listdir(current_path)
     if 'GHRSST-SSTfnd-MUR-GLOB-v02.0-fv04.1' in x and x.startswith(str(year))]
)

# Data
file_paths = [os.path.join(current_path, f) for f in files]

SST = xr.open_mfdataset(file_paths, concat_dim='time', combine='nested').\
    sortby('time').\
    drop_duplicates(dim="time", keep="last").\
    sel(time=slice('2002-01-01', '2023-12-31'))

# Selection
SST_1 = SST.sel(lat=slice(9.5, 11.5), lon=slice(-80, -78))[["analysed_sst"]]

SST_2 = SST.sel(lat=slice(20, 22), lon=slice(-83, -81))[["analysed_sst"]]

SST_3 = SST.sel(lat=slice(10.5, 12.5), lon=slice(-68, -66))[["analysed_sst"]]

SST_4 = SST.sel(lat=slice(11.75, 13.75), lon=slice(-73.25, -71.25))[["analysed_sst"]]

# Export
export_path= "./netcdf_data"
SST_1.to_netcdf(f"{export_path}/polygon_1/data_{year}.nc")
SST_2.to_netcdf(f"{export_path}/polygon_2/data_{year}.nc")
SST_3.to_netcdf(f"{export_path}/polygon_3/data_{year}.nc")
SST_4.to_netcdf(f"{export_path}/polygon_4/data_{year}.nc")

del SST, SST_1, SST_2, SST_3, SST_4

path_SST= "E:/ARCHIVOS E/Data Probability EBUS/MUR/"
year = '2009'
current_path = path_SST

files = sorted(
    [x for x in os.listdir(current_path)
     if 'GHRSST-SSTfnd-MUR-GLOB-v02.0-fv04.1' in x and x.startswith(str(year))]
)

# Data
file_paths = [os.path.join(current_path, f) for f in files]

SST = xr.open_mfdataset(file_paths, concat_dim='time', combine='nested').\
    sortby('time').\
    drop_duplicates(dim="time", keep="last").\
    sel(time=slice('2002-01-01', '2023-12-31'))

# Selection
SST_1 = SST.sel(lat=slice(9.5, 11.5), lon=slice(-80, -78))[["analysed_sst"]]

SST_2 = SST.sel(lat=slice(20, 22), lon=slice(-83, -81))[["analysed_sst"]]

SST_3 = SST.sel(lat=slice(10.5, 12.5), lon=slice(-68, -66))[["analysed_sst"]]

SST_4 = SST.sel(lat=slice(11.75, 13.75), lon=slice(-73.25, -71.25))[["analysed_sst"]]

# Export
export_path= "./netcdf_data"
SST_1.to_netcdf(f"{export_path}/polygon_1/data_{year}.nc")
SST_2.to_netcdf(f"{export_path}/polygon_2/data_{year}.nc")
SST_3.to_netcdf(f"{export_path}/polygon_3/data_{year}.nc")
SST_4.to_netcdf(f"{export_path}/polygon_4/data_{year}.nc")

del SST, SST_1, SST_2, SST_3, SST_4