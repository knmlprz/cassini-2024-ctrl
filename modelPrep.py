#!/usr/bin/env python3

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
import os
from PIL import Image, ImageFile, UnidentifiedImageError

train_dir = 'data/train/'
val_dir = 'data/valid/'

def repair_images(folder_path):
    '''Function to repair corrupted images'''
    print(f"Repairing images in {folder_path}...")
    for image_file in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_file)
        try:
            img = Image.open(image_path)
            img.verify()  # Check for corruption
            img = Image.open(image_path)  # Reopen to load properly
            img.save(image_path)  # Save as repaired
        except (OSError, UnidentifiedImageError) as e:
            print(f"Corrupted image skipped: {image_path}")

# Repair all the images in dataset folders
repair_images("data/train/wildfire/")
repair_images("data/train/nowildfire/")
repair_images("data/valid/nowildfire/")
repair_images("data/valid/wildfire/")
repair_images("data/test/wildfire/")
repair_images("data/test/nowildfire/")

# Data generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)
val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

# Load pre-trained model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
for layer in base_model.layers:
    layer.trainable = False

# Add custom layers to NN
x = Flatten()(base_model.output)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=x)

# Compile model
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Model training
history = model.fit(
    train_generator,
    epochs=1,
    validation_data=val_generator
)

# Evaluate model
loss, accuracy = model.evaluate(val_generator)
print(f'Validation Loss: {loss}, Validation Accuracy: {accuracy}')

# Save model
model.save('wildfire_classifier.h5')
