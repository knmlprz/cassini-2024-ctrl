#!/usr/bin/env python3

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Paths to files
geo_coordinates_path = "geo_coordinates.nc"
gifapar_path = "gifapar.nc"
iwv_path = "iwv.nc"
base_image_path = "streets.png"

# Load data
geo_coordinates = xr.open_dataset(geo_coordinates_path)
gifapar = xr.open_dataset(gifapar_path)
iwv = xr.open_dataset(iwv_path)

latitude = geo_coordinates['latitude'].values
longitude = geo_coordinates['longitude'].values
gifapar_values = gifapar['GIFAPAR'].values
iwv_values = iwv['IWV'].values

# Classification thresholds
safe_gifapar_threshold = 0.4
unsafe_iwv_threshold = 20.0

# Classify safety
safe_areas = (gifapar_values >= safe_gifapar_threshold) & (iwv_values <= unsafe_iwv_threshold)
unsafe_areas = (gifapar_values < safe_gifapar_threshold) & (iwv_values > unsafe_iwv_threshold)
classification = np.zeros_like(gifapar_values)
classification[safe_areas] = 1
classification[unsafe_areas] = -1

# Load base image
base_image = Image.open(base_image_path)

# Overlay plot
plt.figure(figsize=(12, 8))
plt.imshow(base_image, extent=[longitude.min(), longitude.max(), latitude.min(), latitude.max()], alpha=0.7)
plt.pcolormesh(longitude, latitude, classification, cmap="coolwarm", shading="auto", alpha=0.5)
plt.colorbar(label="Classification (-1: Unsafe, 0: Neutral, 1: Safe)")
plt.title("Safety Map Overlay on Street Map")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()
