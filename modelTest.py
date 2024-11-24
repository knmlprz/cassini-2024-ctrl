#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt

model = load_model('wildfire_classifier.h5')

# Load and preprocess the satellite image
def preprocess_image(image_path, target_size=(224, 224)):
    img = load_img(image_path, target_size=target_size)  # Load and resize the image
    img_array = img_to_array(img)  # Convert image to array
    img_array = img_array / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Predict using the model
def predict_image(image_path):
    preprocessed_img = preprocess_image(image_path)
    prediction = model.predict(preprocessed_img)[0][0]  # Get the prediction score
    label = 'Wildfire' if prediction > 0.5 else 'No Wildfire'  # Apply threshold
    confidence = prediction if prediction > 0.5 else 1 - prediction
    return label, confidence

# Visualize the result
def show_image_with_prediction(image_path):
    label, confidence = predict_image(image_path)
    img = load_img(image_path)  # Load the image for display
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"Prediction: {label} ({confidence:.2f} confidence)")
    plt.show()


show_image_with_prediction('output_image.png')
