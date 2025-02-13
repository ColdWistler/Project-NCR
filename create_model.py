from __future__ import print_function
import os
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import TensorBoard

# Paths
BASE_DIR = "/home/soggygenus/Documents/CV/devanagari-character-recognition"
TRAIN_DIR = "/home/soggygenus/Documents/CV/devanagari-character-recognition/train/"
TEST_DIR = "/home/soggygenus/Documents/CV/devanagari-character-recognition/test/"
LOG_DIR = "/home/soggygenus/Documents/CV/devanagari-character-recognition/"

# Data Augmentation
train_datagen = ImageDataGenerator(rescale=1.0/255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1.0/255)

training_set = train_datagen.flow_from_directory(TRAIN_DIR, target_size=(32, 32), batch_size=100, class_mode='categorical')
test_set = test_datagen.flow_from_directory(TEST_DIR, target_size=(32, 32), batch_size=100, class_mode='categorical')

# Model
num_classes = 46
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(32, 32, 3)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="same"),
    Conv2D(64, (3,3), activation='relu'),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="same"),
    Dropout(0.25),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),  # Removed kernel_initializer
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks
tensorboard_callback = TensorBoard(log_dir=LOG_DIR, write_graph=True, write_images=True)

# Training
model.fit(training_set,
          steps_per_epoch=len(training_set),
          epochs=50,
          validation_data=test_set,
          validation_steps=len(test_set),
          callbacks=[tensorboard_callback])

# Save Model
model.save("devnagri_character_model.keras")
