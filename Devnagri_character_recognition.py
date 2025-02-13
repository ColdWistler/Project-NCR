from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import os

# Data Preprocessing
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

training_set = train_datagen.flow_from_directory(
    '/home/soggygenus/Documents/CV/devanagari-character-recognition/train/',
    target_size=(32, 32),
    batch_size=32
)

# Load model
classifier = load_model('/home/soggygenus/Documents/CV/devanagari-character-recognition/devnagri_character_model.keras')
classifier.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Class Indices
classes = training_set.class_indices

# Load Image
image_path = input("Enter image name & path (without .png extension): ") + ".png"
if not os.path.exists(image_path):
    print("Error: File not found!")
    exit()

testImage = cv2.imread(image_path)
testImage = cv2.resize(testImage, (32, 32))
testimage = np.array(testImage) / 255.0  # Normalize
testimage = np.expand_dims(testimage, axis=0)  # Add batch dimension

# Display Image
plt.title("Input Image")
plt.imshow(cv2.cvtColor(testImage, cv2.COLOR_BGR2RGB))
plt.show()

# Predict Class
prediction = np.argmax(classifier.predict(testimage), axis=1)

# Output Result
for key, value in classes.items():
    if value == prediction:
        print("Predicted Class:", key)
