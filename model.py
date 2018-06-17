import sys
import os
import csv
from random import shuffle

import cv2
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Flatten,\
                         Dense,\
                         Lambda,\
                         Conv2D,\
                         MaxPooling2D,\
                         Dropout, \
                         Cropping2D

DATA_DIR = 'data'
IMG_DIR = os.path.join(DATA_DIR, 'IMG')

samples = []
with open(os.path.join(DATA_DIR, 'driving_log.csv')) as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    for line in reader:
        samples.append(line)

train_samples, validation_samples = train_test_split(samples, test_size=0.2)



def fetch_view_angle(batch_sample):
    """
    Conducts Preprocessing on a single data point.
    1. flips original image
    2. adds an offset to steering angle depending upon camera view i.e left, center, right.

    Arguments
    ---------
    batch_sample: array_like
             Elements as [path_center_image, path_left_image, path_right_image, steering_angle, ..]

    Returns
    ---------
    res_images: array_like
            Elements as original and fliped images of each camera view as numpy ndarray.

    res_angles: array_like
            Elements as steering angle of original and fliped images of each camera view as float.
    """
    res_images, res_angles = [], []
    # fetch center angle
    center_angle = float(batch_sample[3])
    viewpoints = ['center', 'left', 'right']
    for idx, view in enumerate(viewpoints):
        filename = os.path.join(IMG_DIR, batch_sample[idx].split('/')[-1])
        image = cv2.imread(filename)
        # Store original image
        res_images.append(image)
        # store fliped image
        res_images.append(cv2.flip(image, 1))
        offset = 0.1

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
    """
    Generates a batch of data on the fly

    Arguments
    ---------
    samples: numpy ndarray
             4 dimensional numpy array of images

    batch_size: int
             Size of the data to be generated

    Returns
    ---------
    4-D numpy ndarray of size(axis = 0) batch_size

    """
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                _images, _angles = fetch_view_angle(batch_sample = batch_sample)
                images.extend(_images)
                angles.extend(_angles)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)


def sanity_check_model():
    """
    Bare Bones model with one no hidden layer i.e flattened input features
    directly connected to output node.

    This model is suppose to be used when building pipeline with minimum focus on model
    performance.


    Returns
    ---------
    keras model

    """
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
    """
    Conventional LeNet model.

    This model is suppose to be used when building insight about the model performance.

    Returns
    ---------
    keras model
    """
    # Initialize model
    model = Sequential()

    # Preprocess incoming data, centered around zero with small standard deviation 
    model.add(Lambda(lambda x: (x - 127)/255, input_shape = (160, 320, 3)))

    # Crop image, removing hood and beyond horizon
    model.add(Cropping2D(cropping = ((70, 25), (0, 0))))

    # First: Convolutional layer
    model.add(Conv2D(6, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # Second: Convolutional layer
    model.add(Conv2D(6, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # Third: Fully Connected layer
    model.add(Flatten())
    model.add(Dense(120))
    model.add(Dropout(0.5))

    # Fourth: Fully Connected layer
    model.add(Dense(84))
    model.add(Dropout(0.5))

    # Fourth: Output layer
    model.add(Dense(1))

    model.compile(loss='mse', optimizer='adam')

    return model


def nvidia():
    """
    Model architeture used by Nvidia for end-to-end human behaviour cloning.

    Reference: https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf

    This is an even powerfull network with 5 Convolutional layers and 3 Fully connected layers.

    Returns
    ---------
    keras model
    """
    # Initialize model
    model = Sequential()

    # Preprocess incoming data, centered around zero with small standard deviation 
    model.add(Lambda(lambda x: (x - 127)/255, input_shape = (160, 320, 3)))

    # Crop image, removing hood and beyond horizon
    model.add(Cropping2D(cropping = ((70, 25), (0, 0))))

    # First: Convolutional layer
    model.add(Conv2D(24, (5, 5), strides = (2, 2), activation='relu'))
    model.add(Dropout(0.25))
    #model.add(BatchNormalization(axis = 1))

    # Second: Convolutional layer
    model.add(Conv2D(36, (5, 5), strides = (2, 2), activation='relu'))
    model.add(Dropout(0.25))
    #model.add(BatchNormalization(axis = 1))

    # Third: Convolutional layer
    model.add(Conv2D(48, (5, 5), strides = (2, 2), activation='relu'))
    model.add(Dropout(0.25))
    #model.add(BatchNormalization(axis = 1))

    # Fourth: Convolutional layer
    model.add(Conv2D(64, (3, 3), strides = (1, 1), activation='relu'))
    model.add(Dropout(0.25))
    #model.add(BatchNormalization(axis = 1))

    # Fifth: Convolutional layer
    model.add(Conv2D(64, (3, 3), strides = (1, 1), activation='relu'))
    model.add(Dropout(0.25))
    #model.add(BatchNormalization(axis = 1))

    model.add(Flatten())
    # Sixth: Fully Connected layer
    model.add(Dense(100))
    model.add(Dropout(0.5))

    # Seventh: Fully Connected layer
    model.add(Dense(50))
    model.add(Dropout(0.5))

    # Eigth: Fully Connected layer
    model.add(Dense(10))
    model.add(Dropout(0.5))

    # Ninth: Output layer
    model.add(Dense(1))

    model.compile(loss='mse', optimizer='adam')

    return model

def get_model(name = 'sanity_check'):
    """
    Return appropriate model

    Arguments
    ---------
    name: string
         Name of the model to be trained

    Returns
    ---------
    Keras model
    """

    if name == 'sanity_check':
        return sanity_check_model()

    if name ==  'LeNet':
        return LeNet()

    if name ==  'nvidia':
        return nvidia()

batch_size = 64
train_generator = generator(train_samples, batch_size = batch_size)
validation_generator = generator(validation_samples, batch_size = batch_size)

# Final Model Architecture to be used
model_name = 'nvidia'

print("Traning samples : {} | Validation samples : {}"\
      .format(3*2*len(train_samples), 3*2*len(validation_samples)))
print(model_name)

model = get_model(name = model_name)
history_object = model.fit_generator(train_generator, steps_per_epoch= \
            2*3*len(train_samples)//batch_size, validation_data=validation_generator, \
            validation_steps=3*2*len(validation_samples)//batch_size, epochs=5)

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.imsave('post_training_analysis.png')

model.save('model_{}.h5'.format(model_name))
