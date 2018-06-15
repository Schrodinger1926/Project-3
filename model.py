import os
import csv
from random import shuffle

import cv2
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Flatten, Dense

DATA_DIR = 'data'

samples = []
with open(os.path.join(DATA_DIR, 'driving_log.csv')) as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    for line in reader:
        samples.append(line)

train_samples, validation_samples = train_test_split(samples, test_size=0.2)

print("Traning samples : {} | Validation samples : {}"\
      .format(len(train_samples), len(validation_samples)))

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = 'data/IMG/'+batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)


def sanity_check_model():
    # Initialize model
    model = Sequential()
    # Preprocess incoming data, centered around zero with small standard deviation 
    model.add(Flatten(input_shape = (160, 320, 3)))
    #model.add(Lambda(lambda x: x/127.5 - 1.)
    model.add(Dense(1))
    # Comple model
    model.compile(loss='mse', optimizer='adam')

    return model

train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)
ch, row, col = 3, 160, 320  # Trimmed image format

model = sanity_check_model()
model.fit_generator(train_generator, samples_per_epoch= \
            len(train_samples), validation_data=validation_generator, \
            nb_val_samples=len(validation_samples), nb_epoch=3)

model.save('model.h5')
