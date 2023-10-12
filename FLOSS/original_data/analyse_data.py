import netCDF4
import pandas as pd
import numpy as np
import xarray as xr

data_file_names = ['flossii.021003.nc', 'flossii.021004.nc']
data_file_paths = ['data_files/'+file for file in data_file_names]

ds = xr.open_mfdataset('single_files/*.nc', combine='nested', concat_dim = 'time')
# ds.to_netcdf('nc_combined.nc') # Export netcdf file
df = ds.to_dataframe()
print(df['station'])