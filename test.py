#!/usr/bin/env python3

import os
from dotenv import load_dotenv
from sentinelhub import SHConfig, SentinelHubRequest, MimeType, CRS, BBox, bbox_to_dimensions, DataCollection, SentinelHubDownloadClient, DataCollection, DownloadRequest
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Resizing
from skimage.util import view_as_blocks
import matplotlib.pyplot as plt

load_dotenv()

# Step 1: Sentinel Hub Configuration
config = SHConfig()
config.sh_client_id = os.getenv("CLIENT_ID")  # Replace with your Sentinel Hub client ID
config.sh_client_secret = os.getenv("CLIENT_SECRET")  # Replace with your Sentinel Hub client secret


# Define Area of Interest (AOI) and resolution
aoi_bbox = BBox(bbox=(-5.0, 40.0, -4.8, 40.2), crs=CRS.WGS84)  # Example AOI
resolution = 300  # Sentinel-3 OLCI resolution in meters

# Step 2: Define SentinelHubRequest for Sentinel-3 OLCI data
def get_sentinel3_data(aoi_bbox, time_interval):
    size = bbox_to_dimensions(aoi_bbox, resolution=resolution)

    request = SentinelHubRequest(
        evalscript="""
            // Get Sentinel-3 OLCI bands (e.g., B08, B04, B03)
            function setup() {
                return {
                    input: ["B08", "B04", "B03"], // NIR, Red, Green
                    output: { bands: 3 }
                };
            }

            function evaluatePixel(sample) {
                return [sample.B08, sample.B04, sample.B03];
            }
        """,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL3_OLCI,
                time_interval=time_interval
            )
        ],
        responses=[
            SentinelHubRequest.output_response('default', MimeType.TIFF)
        ],
        bbox=aoi_bbox,
        size=size,
        config=config
    )

    response = request.get_data()
    return np.array(response[0])  # Returns a NumPy array of shape (H, W, 3)

# Example Usage
data = get_sentinel3_data(aoi_bbox, time_interval=("2023-01-01", "2023-01-15"))
print(f"Downloaded data shape: {data.shape}")
