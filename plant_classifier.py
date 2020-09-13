import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
#from tensorflow.keras.preprocessing.image import img_to_array, load_img
#import random
import os

#Training dataset
TRAINING_DIR = "Dataset/Train" 

training_datagen = ImageDataGenerator(
    rescale = 1./255,
    rotation_range=360,
    width_shift_range=1,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.5,
    horizontal_flip=True,
    fill_mode='nearest')

VALIDATION_DIR = "Dataset/Validation"

validation_datagen = ImageDataGenerator(
    rescale = 1./255,
    rotation_range=360,
    width_shift_range=1,
    height_shift_range=1,
    shear_range=1,
    zoom_range=1,
    horizontal_flip=True,
    fill_mode='nearest')


#Train dataset + image gen + catagorical or binary
train_generator = training_datagen.flow_from_directory(
    TRAINING_DIR,
    target_size=(255,255),
    class_mode='categorical')

validation_generator = validation_datagen.flow_from_directory(
    VALIDATION_DIR,
    target_size=(255,255),
    class_mode='categorical')

#CNN (ppt)/ DNN
model = tf.keras.models.Sequential([
    #relu in ppt
    # This is the first convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(255, 255, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The second convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The third convolution
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The fourth convolution
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(512, activation='relu'),
    #Enter number of classes here
    tf.keras.layers.Dense(9, activation='softmax')
])

model.compile(loss = 'categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

history = model.fit_generator(train_generator, epochs=10, validation_data = validation_generator, verbose = 1)

model.save("rps_tomato.h5")


acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.figure()
plt.show()