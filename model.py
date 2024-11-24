#!/usr/bin/env python3

import os
from dotenv import load_dotenv
from sentinelhub import SHConfig, SentinelHubRequest, MimeType, CRS, BBox, bbox_to_dimensions, DataCollection
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

# Step 3: Retrieve and preprocess Sentinel-3 OLCI data
data = get_sentinel3_data(aoi_bbox, time_interval=("2023-01-01", "2023-01-15"))
data_normalized = data / 10000.0  # Normalize pixel values to [0, 1]

# Visualize the retrieved image
plt.imshow(data_normalized)
plt.title("Sentinel-3 Data (NIR-Red-Green Composite)")
plt.axis("off")
# plt.show()

# Step 4: Split image into tiles
def create_tiles(image, tile_size):
    h, w, c = image.shape
    h_adjusted = (h // tile_size) * tile_size
    w_adjusted = (w // tile_size) * tile_size
    image_cropped = image[:h_adjusted, :w_adjusted, :]
    tiles = view_as_blocks(image_cropped, block_shape=(tile_size, tile_size, c))
    return tiles.reshape(-1, tile_size, tile_size, c)

# Generate tiles of size 128x128
tiles = create_tiles(data_normalized, tile_size=128)
print(f"Number of tiles: {tiles.shape[0]}")

# Step 5: Generate labels
labels = np.array([1 if np.mean(tile) > 0.3 else 0 for tile in tiles])
assert len(labels) == tiles.shape[0], "Number of labels must match the number of samples"

# Step 6: Train/Test Split
split_idx = int(0.8 * len(tiles))
train_data = tiles[:split_idx]
test_data = tiles[split_idx:]
train_labels = labels[:split_idx]
test_labels = labels[split_idx:]

assert train_data.shape[0] == train_labels.shape[0], "Mismatch in train data and labels"
assert test_data.shape[0] == test_labels.shape[0], "Mismatch in test data and labels"

# Step 7: Define the CNN Model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(128, 128, 3)),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Binary classification
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Step 8: Train the Model
history = model.fit(
    train_data, train_labels,
    validation_split=0.2,
    epochs=10,
    batch_size=32
)

# Step 9: Evaluate the Model
test_loss, test_accuracy = model.evaluate(test_data, test_labels)
print(f"Test Accuracy: {test_accuracy:.2f}")

# Step 10: Save the Model
model.save("sentinel3_binary_classifier.h5")
