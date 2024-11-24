#!/usr/bin/env python3

'''Download satellite image from SentinelHub API chosen by coordinates'''

# Utilities
import os
import matplotlib.pyplot as plt
import pandas as pd
from dotenv import load_dotenv

# EDC libraries
# from edc import setup_environment_variables

from sentinelhub import (
    SHConfig,
    DataCollection,
    SentinelHubCatalog,
    SentinelHubRequest,
    SentinelHubStatistical,
    BBox,
    bbox_to_dimensions,
    CRS,
    MimeType,
    Geometry,
)

from utils import plot_image

load_dotenv()

config = SHConfig()
config.sh_client_id = os.environ["CLIENT_ID"]
config.sh_client_secret = os.environ["CLIENT_SECRET"]

# aoi_coords_wgs84 = [21.979018,50.017412,21.987065,50.020390]
aoi_coords_wgs84 = [21.8615417, 49.9361266, 22.0931153, 50.0925333]


resolution = 10
aoi_bbox = BBox(bbox=aoi_coords_wgs84, crs=CRS.WGS84)
aoi_size = bbox_to_dimensions(aoi_bbox, resolution=resolution)

# print(f"Image shape at {resolution} m resolution: {aoi_size} pixels")


catalog = SentinelHubCatalog(config=config)

aoi_bbox = BBox(bbox=aoi_coords_wgs84, crs=CRS.WGS84)
time_interval = "2022-05-01", "2022-05-20"

search_iterator = catalog.search(
    DataCollection.SENTINEL2_L2A,
    bbox=aoi_bbox,
    time=time_interval,
    fields={"include": ["id", "properties.datetime"], "exclude": []},
)

results = list(search_iterator)
# print("Total number of results:", len(results))

# results


evalscript_true_color = """
    //VERSION=3

    function setup() {
        return {
            input: [{
                bands: ["B02", "B03", "B04"]
            }],
            output: {
                bands: 3
            }
        };
    }

    function evaluatePixel(sample) {
        return [sample.B04, sample.B03, sample.B02];
    }
"""

request_true_color = SentinelHubRequest(
    evalscript=evalscript_true_color,
    input_data=[
        SentinelHubRequest.input_data(
            data_collection=DataCollection.SENTINEL2_L2A,
            time_interval=("2022-05-01", "2022-05-20")        )
    ],
    responses=[SentinelHubRequest.output_response("default", MimeType.PNG)],
    bbox=aoi_bbox,
    size=aoi_size,
    config=config,
)

true_color_imgs = request_true_color.get_data()

print(
    f"Returned data is of type = {type(true_color_imgs)} and length {len(true_color_imgs)}."
)
print(
    f"Single element in the list is of type {type(true_color_imgs[-1])} and has shape {true_color_imgs[-1].shape}"
)



image = true_color_imgs[0]
print(f"Image type: {image.dtype}")

# plot function
# factor 1/255 to scale between 0-1
# factor 3.5 to increase brightness
plot_image(image, factor=3.5 / 255, clip_range=(0, 1))



plt.savefig("output_image.png", bbox_inches="tight", pad_inches=0)
plt.show()
