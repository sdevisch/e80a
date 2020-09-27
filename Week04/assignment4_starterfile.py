import os
import sys
import numpy as np
from tensorflow.keras import models, layers, callbacks
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist
from matplotlib import pyplot as plt

# Change this to the location of the database directories
DB_DIR = os.path.dirname(os.path.realpath(__file__))

# Import databases
sys.path.insert(1, DB_DIR)
from db_utils import get_imdb_dataset, get_speech_dataset

def Secure_Voice_Channel(func):
    """Define Secure_Voice_Channel decorator."""
    def execute_func(*args, **kwargs):
        print('Established Secure Connection.')
        returned_value = func(*args, **kwargs)
        print("Ended Secure Connection.")

        return returned_value

    return execute_func

@Secure_Voice_Channel
def generic_vns_function(input_dim, number_dense_layers, classes, units, lr):
    """Generic Deep Learning Model generator."""
    model = models.Sequential()
    model.add(layers.Conv2D(64, (4,4), activation='relu'))
    model.add(layers.MaxPool2D(2,2))
    model.add(layers.Flatten())
    for i in range(number_dense_layers):
        model.add(layers.Dense(units=units, input_dim=input_dim,
                  kernel_initializer='normal', activation='relu'))

    model.add(layers.Dense(classes, kernel_initializer='normal',
              activation='softmax'))
    opt = Adam(lr=lr)
    model.compile(loss='categorical_crossentropy', optimizer=opt,
                  metrics=['accuracy'])
    return model

def train_model(model, epochs, batch_size, X_train, y_train, X_test, y_test):
    """Generic Deep Learning Model training function."""
    cb = [callbacks.EarlyStopping(monitor='val_loss', patience=3)]
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs,
              batch_size=batch_size, verbose=1, callbacks=cb)
    scores = model.evaluate(X_test, y_test, verbose=2)

    print("Baseline Error: %.2f%%" % (100-scores[1]*100))

    return model

def choose_dataset(dataset_type):
    """Select dataset based on string variable."""
    if dataset_type == "nlp":
        return get_imdb_dataset(dir=DB_DIR)
    elif dataset_type == "computer_vision":
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
    elif dataset_type == "speech_recognition":
        (X_train, y_train), (X_test, y_test), (_, _) = get_speech_dataset()
    else:
        raise ValueError("Couldn't find dataset.")

    (X_train, X_test) = normalize_dataset(dataset_type, X_train, X_test)

    (X_train, y_train), (X_test, y_test) = reshape_dataset(X_train, y_train, X_test, y_test)

    return (X_train, y_train), (X_test, y_test)

def normalize_dataset(string, X_train, X_test):
    """Normalize speech recognition and computer vision datasets."""
    if string is "computer vision":
        X_train = X_train / 255
        X_test = X_test / 255
    else:
        mean = np.mean(X_train)
        std = np.std(X_train)
        X_train = (X_train-std)/mean
        X_test = (X_test-std)/mean

    return (X_train, X_test)

def reshape_dataset(X_train, y_train, X_test, y_test):
    """Reshape Computer Vision and Speech datasets."""

    num_pixels = X_test.shape[1]*X_test.shape[2]

    # TODO: Apply transformations to data
    X_train, X_test = add_padding(X_train), add_padding(X_test)
    X_train, X_test = rotate_array(X_train), rotate_array(X_test)


    # TODO: Change Reshape Function to the format (num_samples, x_axis, y_axis, channels)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1).astype('float32')
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1).astype('float32')

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    return (X_train, y_train), (X_test, y_test)

def main():

    # Hyperparameters
    layers = 2
    layer_units = 1000
    epochs = 1
    batch_size = 200
    lr = 0.0001

    # Dataset : "computer_vision"
    dataset = "computer_vision"

    # Import Datasets
    (X_train, y_train), (X_test, y_test) = choose_dataset(dataset)

    # Generate and train model
    # TODO: Change inputs to generic_vns_function
    model = generic_vns_function(X_train.shape[1], layers, y_train.shape[1], layer_units, lr)
    trained_model = train_model(model, epochs, batch_size, X_train, y_train, X_test, y_test)

    # Save model to h5 file
    trained_model.save('models/model_%s_a4.h5' % dataset)

    return None

######################DATA AUGMENTATION FUNCTIONS###############################
################################################################################

import matplotlib.pyplot as plt
import numpy as np
from random import random, seed, randint, uniform
import cv2
from scipy.ndimage.interpolation import rotate, zoom


def add_padding(X, padding=10):
    """Add padding to images in array, by changing image size"""
    new_X = []
    for img in X:
        new_image = np.zeros((img.shape[0]+padding*2,img.shape[0]+padding*2))
        new_image[padding:padding+img.shape[0],padding:padding+img.shape[1]] = img
        new_X.append(new_image)
    return np.asarray(new_X)

def move_array(X, transform_range=5):
    """Transform X (image array) with a random move within range."""
    new_X = []
    for img in X:
        moved_x = randint(-transform_range, transform_range)
        moved_y = randint(-transform_range, transform_range)
        translation_matrix = np.float32([[1,0,moved_x], [0,1,moved_y]])
        moved_image = cv2.warpAffine(img, translation_matrix, (img.shape[0], img.shape[0]))
        new_X.append(moved_image)

    return np.asarray(new_X)

def rotate_array(X, angle_range=60):
    """Rotate X (image array) with a random angle within angle range."""
    new_X = []
    for img in X:
        angle = random()*angle_range*2-angle_range
        new_image = rotate(img, angle=angle, reshape=False)
        new_X.append(new_image)
    return np.asarray(new_X)

def zoom_array(X, zoom_range_min=0.6, zoom_range_max=1):
    """Zoom X (image array) with a random zoom within zoom range."""
    new_X = []
    for img in X:
        zoom = uniform(zoom_range_min,zoom_range_max)
        zoomed_image = clipped_zoom(img, zoom)
        new_X.append(zoomed_image)
    return np.asarray(new_X)

def clipped_zoom(img, zoom_factor, **kwargs):
    """Zoom while clippig image to keep array size."""

    # Code inspired on ali_m's response in Stack Overflow issue:
    # https://stackoverflow.com/questions/37119071/scipy-rotate-and-zoom-an-image-without-changing-its-dimensions
    h, w = img.shape[:2]

    # For multichannel images we don't want to apply the zoom factor to the RGB
    # dimension, so instead we create a tuple of zoom factors, one per array
    # dimension, with 1's for any trailing dimensions after the width and height.
    zoom_tuple = (zoom_factor,) * 2 + (1,) * (img.ndim - 2)

    # Zooming out
    if zoom_factor < 1:

        # Bounding box of the zoomed-out image within the output array
        zh = int(np.round(h * zoom_factor))
        zw = int(np.round(w * zoom_factor))
        top = (h - zh) // 2
        left = (w - zw) // 2

        # Zero-padding
        out = np.zeros_like(img)
        out[top:top+zh, left:left+zw] = zoom(img, zoom_tuple, **kwargs)

    # Zooming in
    elif zoom_factor > 1:

        # Bounding box of the zoomed-in region within the input array
        zh = int(np.round(h / zoom_factor))
        zw = int(np.round(w / zoom_factor))
        top = (h - zh) // 2
        left = (w - zw) // 2

        out = zoom(img[top:top+zh, left:left+zw], zoom_tuple, **kwargs)
        # `out` might still be slightly larger than `img` due to rounding, so
        # trim off any extra pixels at the edges
        if out.shape[0] < h:
            out = add_padding([out],padding=1)[0]

        trim_top = ((out.shape[0] - h) // 2)
        trim_left = ((out.shape[1] - w) // 2)
        out = out[trim_top:trim_top+h, trim_left:trim_left+w]
    # If zoom_factor == 1, just return the input array
    else:
        out = img

    return out

if __name__ == '__main__':
    main()
