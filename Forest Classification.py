#frameworks
import tensorflow as tf
import kaggle

#processing
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

#visualizations
import numpy as np
import pandas as pd

#models, layers, and optimizers
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.optimizers import Adam

#datasets
path = "C:/Users/parsa/.kaggle/input/forest-vs-desert"
class_names = ["Desert", "Forest"]
img_path = "C:/Users/parsa/Desktop/test_image.jpg"  # Replace with your test image path


def Predict_Image(prediction_img, model):
    img = image.load_img(prediction_img, target_size=(256, 256, 3))
    model.predict(img)
    predictions = model.predict(img)
    prediction = model.predict(img) 
    
    predicted_class = 1 if prediction[0][0] > 0.5 else 0 
    
    return class_names[predicted_class], prediction[0][0] 




#Innitialize the image data generator
train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip = True,
    validation_split = 0.2)

# Load dataset
training_set = train_datagen.flow_from_directory(
    path,
    target_size=(256, 256),
    color_mode='rgb',
    batch_size=32,
    class_mode='binary',  
    shuffle=True,
    subset='training'
)

testing_set = train_datagen.flow_from_directory(
    path,
    target_size=(256, 256),
    color_mode='rgb',
    batch_size=32,
    class_mode='binary',  
    shuffle=True,
    subset='validation'
)

# Fix model architecture
model = Sequential()
model.add(Conv2D(32, 3, activation='relu', input_shape=(256, 256, 3)))
model.add(MaxPooling2D(2, 2))
model.add(Conv2D(64, 3, activation='relu'))
model.add(MaxPooling2D(2, 2))
model.add(Conv2D(128, 3, activation='relu'))
model.add(MaxPooling2D(2, 2))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  # Use 3 neurons and softmax for 3 classes

# Compile with categorical crossentropy
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(training_set, validation_data = testing_set, epochs = 4, verbose = 1)

print("The input image is: ", Predict_Image(img_path, model))  

loss, acc = model.evaluate(training_set, verbose = 0)