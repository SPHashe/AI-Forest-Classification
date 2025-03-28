#frameworks
import tensorflow as tf
import kaggle

#processing
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

#visualizations
import pandas as pd

#models, layers, and optimizers
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.optimizers import Adam

#datasets
path = "\Users\parsa\.kaggle\input\forest-vs-desert"

#Innitialize the image data generator
train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip = True,
    validation_split = 0.2)

training_set, testining_set = train_datagen.flow_from_directory(
    path,
    target_size = (256, 256),
    color_mode = 'rgb',
    batch_size = 32,
    class_mode = 'binary',
    shuffle = True,
    subset = 'training')

#training_set.labels.astype('boolean')