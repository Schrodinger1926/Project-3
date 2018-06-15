import os
import csv
from random import shuffle

import cv2
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D, MaxPooling2D, Dropout, Cropping2D

DATA_DIR = 'data'
IMG_DIR = os.path.join(DATA_DIR, 'IMG')

samples = []
with open(os.path.join(DATA_DIR, 'driving_log.csv')) as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    for line in reader:
        samples.append(line)

train_samples, validation_samples = train_test_split(samples, test_size=0.2)

print("Traning samples : {} | Validation samples : {}"\
      .format(len(train_samples), len(validation_samples)))


def fetch_view_angle(batch_sample, viewpoints):
    res_images, res_angles = [], []
    # fetch center angle
    center_angle = float(batch_sample[3])
    for idx, view in enumerate(viewpoints):
        filename = os.path.join(IMG_DIR, batch_sample[idx].split('/')[-1])
        image = cv2.imread(filename)
        # Store original image
        res_images.append(image)
        # store fliped image
        res_images.append(cv2.flip(image, 1))
        offset = 0.2

        if view == 'center':
            # Store angles
            res_angles.append(center_angle)
            # Store flip angle
            res_angles.append(-center_angle)

        if view == 'left':
            # Store angle
            res_angles.append(center_angle + offset)
            # Store flip angle
            res_angles.append(-(center_angle + offset))

        if view == 'right':
            # Store angle
            res_angles.append(center_angle - offset)
            # Store fliped angle
            res_angles.append(-(center_angle - offset))

    return res_images, res_angles


def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                """
                name = 'data/IMG/'+batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)
                """
                _images, _angles = fetch_view_angle(batch_sample = batch_sample,
                                                viewpoints = ['center', 'left', 'right'])
                images.extend(_images)
                angles.extend(_angles)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)


def sanity_check_model():
    # Initialize model
    model = Sequential()
    # Preprocess incoming data, centered around zero with small standard deviation 
    model.add(Flatten(input_shape = (160, 320, 3)))
    # Normalization
    model.add(Lambda(lambda x: (x - 127)/127))
    # Fully connected layer 
    model.add(Dense(1))
    # Comple model
    model.compile(loss='mse', optimizer='adam')

    return model

def LeNet():
    model = Sequential()
    model.add(Lambda(lambda x: (x - 127)/255, input_shape = (160, 320, 3)))
    model.add(Cropping2D(cropping = ((70, 25), (0, 0))))

    model.add(Conv2D(6, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.25))

    model.add(Conv2D(6, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(120))
    #model.add(Dropout(0.5))

    model.add(Dense(84))
    #model.add(Dropout(0.5))

    model.add(Dense(1))

    model.compile(loss='mse', optimizer='adam')

    return model


def nvidia():
    model = Sequential()
    model.add(Lambda(lambda x: (x - 127)/255, input_shape = (160, 320, 3)))
    model.add(Cropping2D(cropping = ((70, 25), (0, 0))))

    model.add(Conv2D(24, (5, 5), strides = (2, 2), activation='relu'))
    model.add(Dropout(0.25))

    model.add(Conv2D(36, (5, 5), strides = (2, 2), activation='relu'))
    model.add(Dropout(0.25))

    model.add(Conv2D(48, (5, 5), strides = (2, 2), activation='relu'))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), strides = (1, 1), activation='relu'))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), strides = (1, 1), activation='relu'))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dropout(0.5))

    model.add(Dense(50))
    model.add(Dropout(0.5))

    model.add(Dense(1))

    model.compile(loss='mse', optimizer='adam')

    return model

def get_model(name = 'sanity_check'):
    if name == 'sanity_check':
        return sanity_check_model()

    if name ==  'LeNet':
        return LeNet()

    if name ==  'nvidia':
        return nvidia()

batch_size = 64
train_generator = generator(train_samples, batch_size = batch_size)
validation_generator = generator(validation_samples, batch_size = batch_size)

model_name = 'nvidia'
model = get_model(name = model_name)
model.fit_generator(train_generator, steps_per_epoch= \
            2*3*len(train_samples)//batch_size, validation_data=validation_generator, \
            validation_steps=len(validation_samples)//batch_size, epochs=3)

model.save('model_{}.h5'.format(model_name))
