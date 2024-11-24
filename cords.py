#!/usr/bin/env python3

import netCDF4 as nc

# Path to the uploaded file
file_path = 'geo_coordinates.nc'

# Open the NetCDF file
dataset = nc.Dataset(file_path, mode='r')

# Get the list of variables to check for WGS84 coordinates
variables = dataset.variables.keys()

# Extract and print details of variables
for var in variables:
    print(f"Variable: {var}")
    print(f"Dimensions: {dataset.variables[var].dimensions}")
    print(f"Attributes: {dataset.variables[var].ncattrs()}")
    print()

dataset.close()
