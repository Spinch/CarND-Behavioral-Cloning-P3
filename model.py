
import pandas as pd
import cv2
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import sklearn
from keras.models import Model, Sequential
from keras.layers.core import Dense, Flatten, Dropout
from keras.layers import Input, Convolution2D, MaxPooling2D, Lambda
from keras.layers.convolutional import Cropping2D, ZeroPadding2D
from keras.callbacks import ModelCheckpoint, EarlyStopping

# from keras.utils.visualize_util import plot
# from keras.utils import plot_model

# Load the data; if loadImages is False, only image filenames will be loaded for future use in generator
def loadData(loadImages = True):
    dataDir = './D'
    X_train = []
    y_train = []
    W = []
    for subDir in os.listdir(dataDir):
        dataCSVFile = os.path.join(dataDir, subDir, 'driving_log.csv')
        data = pd.read_csv(dataCSVFile, ',').values
        for row in data:
            path = os.path.join(dataDir, subDir, 'IMG')
            if loadImages:
                y = row[3]
                x = cv2.imread(os.path.join(path, os.path.split(row[0])[1]))
                # Load base image
                # if prepocessOneData(x, y):

                w = dataProb(y)

                # Load base image
                X_train.append(x)
                y_train.append(y)
                W.append(w)
                # Add flipped image
                X_train.append(np.fliplr(x))
                y_train.append(y*(-1.))
                W.append(w)

                # add image from left camera
                y = row[3] + 0.2*np.random.random()
                if (y > 1.):
                    continue
                x = cv2.imread(os.path.join(path, os.path.split(row[1])[1]))
                X_train.append(x)
                y_train.append(y)
                W.append(w)
                # X_train.append(np.fliplr(x))
                # y_train.append(y * (-1.))

                # add image from right camera
                y = row[3] - 0.2*np.random.random()
                if (y < -1.):
                    continue
                x = cv2.imread(os.path.join(path, os.path.split(row[2])[1]))
                X_train.append(x)
                y_train.append(y)
                W.append(w)
                # X_train.append(np.fliplr(x))
                # y_train.append(y * (-1.))
            else:
                X_train.append(np.append(row, path))
                # y_train.append(row[3])

    return np.array(X_train), np.array(y_train), np.array(W)

def analyseData(X, y, w):
    plt.hist(y, 99, weights=w)
    plt.show()
    pass

def dataProb(a):
    if np.abs(a) < 0.01:
        return 0.5
    return 1.

def prepocessOneData(bx, by):
    # Here we remove 0.5 images with straight wheel angle
    if np.abs(by) < 0.01:
        if np.random.random() > 0.4:
            return False
    return True

def generator(samples, batch_size=128):
    num_samples = len(samples)
    while 1:
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = os.path.join(batch_sample[7], os.path.split(batch_sample[0])[1])
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)
                images.append(np.fliplr(center_image))
                angles.append(center_angle*(-1.))

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield (X_train, y_train)

# create NN Keras model
def createModel(input_shape):
    # print(input_shape)
    model = Sequential()

    # cropping and normalization layers
    model.add(Cropping2D(cropping=((60,20), (0,0)), input_shape=input_shape))
    model.add(Lambda(lambda x: (x / 255.0) - 0.5))

    # activation functions for convolution and dense layers
    actc = 'relu'
    actf = 'relu'

    model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation=actc))
    # ZeroPadding2D(())
    # model.add(MaxPooling2D())
    # model.add(Dropout(0.5))

    model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation=actc))
    # model.add(MaxPooling2D())
    model.add(Dropout(0.5))

    model.add(Convolution2D(48, 5, 5, subsample=(2,2), activation=actc))
    # model.add(MaxPooling2D())
    model.add(Dropout(0.5))

    model.add(Convolution2D(64, 3, 3, subsample=(1,1), activation=actc))
    # model.add(MaxPooling2D())
    model.add(Dropout(0.5))

    model.add(Convolution2D(64, 3, 3, subsample=(1,1), activation=actc))
    # model.add(MaxPooling2D())
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(128, activation=actf))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation=actf))
    model.add(Dropout(0.5))
    model.add(Dense(1))

    return model

if __name__ == '__main__':
    X_train, y_train, W = loadData()

    analyseData(X_train, y_train, W)13
    # sys.exit(0)

    # samples, _ = loadData(False)

    # get image shape
    try:
        shape = X_train.shape[1:]
    except NameError:
        shape = cv2.imread(os.path.join(samples[0][7], os.path.split(samples[0][0])[1])).shape

    # Create and compile model
    model = createModel(shape)
    model.compile(optimizer='Adam', loss='mse')

    print(model.summary())
    # plot(model, to_file='model.png')
    # sys.exit(0)

    X_train, X_valid, y_train, y_valid, W_train, W_valid = sklearn.model_selection.train_test_split(X_train, y_train, W)

    # checkpoint to save model after each epoch
    checkpoint = ModelCheckpoint('s2Model-{epoch:02d}.h5')
    # earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')

    # Train model
    # model.fit(X_train, y_train, batch_size=128, nb_epoch=10, validation_split=0.2, shuffle=True, callbacks=[checkpoint])
    model.fit(X_train, y_train, batch_size=128, nb_epoch=10, validation_data=(X_valid, y_valid), sample_weight=W_train,
              shuffle=True, callbacks=[checkpoint])

    # train_samples, validation_samples = train_test_split(samples, test_size=0.2)
    # train_generator = generator(train_samples, batch_size=128)
    # validation_generator = generator(validation_samples, batch_size=128)
    # model.fit_generator(train_generator, samples_per_epoch=len(train_samples), validation_data=validation_generator,
    #                     nb_val_samples = len(validation_samples), nb_epoch = 5, callbacks=[checkpoint])

    # save final model
    model.save('model.h5')