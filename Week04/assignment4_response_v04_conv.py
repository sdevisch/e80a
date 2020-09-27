import os
import sys
import numpy as np
from tensorflow import keras
from tensorflow.keras import models, layers, callbacks
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist

# Change this to the location of the database directories
DB_DIR = os.path.dirname(os.path.realpath("/Users/stevendevisch/E80a/installation_guide"))
print(DB_DIR)
# Import databases
sys.path.insert(1, DB_DIR)
# from db_utils import get_imdb_dataset, get_speech_dataset

def Secure_Voice_Channel(func):
    """Define Secure_Voice_Channel decorator."""
    def execute_func(*args, **kwargs):
        print('Established Secure Connection.')
        returned_value = func(*args, **kwargs)
        print("Ended Secure Connection.")

        return returned_value

    return execute_func

@Secure_Voice_Channel
def generic_vns_function(input_dim, number_dense_layers, classes, lr):
    """Generic Deep Learning Model generator."""
    model = models.Sequential(
    [
        keras.Input(shape=input_dim),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(classes, activation="softmax"),
    ]
    )
    opt = Adam(lr=lr)
    model.compile(loss='categorical_crossentropy', optimizer=opt,
                  metrics=['accuracy'])
    return model

def train_model(model, epochs, batch_size, x_train, y_train, x_test, y_test):
    """Generic Deep Learning Model training function."""
    cb = [callbacks.EarlyStopping(monitor='val_loss', patience=3)]
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs,
              batch_size=batch_size, verbose=1, callbacks=cb)
    scores = model.evaluate(x_test, y_test, verbose=2)

    print("Baseline Error: %.2f%%" % (100-scores[1]*100))

    return model

def choose_dataset(dataset_type, transformation_type="none"):
    """Select dataset based on string variable."""
    if dataset_type == "nlp":
        return get_imdb_dataset(dir=DB_DIR)
    elif dataset_type == "computer_vision":
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    elif dataset_type == "speech_recognition":
        (x_train, y_train), (x_test, y_test), (_, _) = get_speech_dataset()
    else:
        raise ValueError("Couldn't find dataset.")

    (x_train, x_test) = transform_dataset(dataset_type, x_train, x_test, transformation_type)

    (x_train, x_test) = normalize_dataset(dataset_type, x_train, x_test)

    (x_train, y_train), (x_test, y_test) = reshape_dataset(x_train, y_train, x_test, y_test)

    return (x_train, y_train), (x_test, y_test)

def transform_dataset(string, x_train, x_test, transformation_type):
    """Normalize speech recognition and computer vision datasets."""
    if string is "computer_vision":
        if transformation_type is "none":
            print("not applying any transformations")
        if transformation_type is "rotate":
            print("adding rotated pictures")
            x_train = rotate_array(x_train)
            x_test = rotate_array(x_test)
        if transformation_type is "zoom":
            print("adding zoomed pictures")
            x_train = zoom_array(x_train)
            x_test = zoom_array(x_test)
        if transformation_type is "translate":
            print("adding translated pictures")
            x_train = move_array(x_train)
            x_test = move_array(x_test)
        if transformation_type is "all":
            print("adding all transformations to pictures")
            x_train = rotate_array(x_train)
            x_test = rotate_array(x_test)
            x_train = zoom_array(x_train)
            x_test = zoom_array(x_test)
            x_train = move_array(x_train)
            x_test = move_array(x_test)
        # make sure each pixel has a value
        x_train = add_padding(x_train)
        x_test = add_padding(x_test)
    else:
        print("not implemented yet")


    return (x_train, x_test)


def normalize_dataset(string, x_train, x_test):
    """Normalize speech recognition and computer vision datasets."""
    if string is "computer_vision":
        x_train = x_train.astype("float32") / 255
        x_test = x_test.astype("float32") / 255
        # Make sure images have shape (28, 28, 1)
        x_train = np.expand_dims(x_train, -1)
        x_test = np.expand_dims(x_test, -1)
    else:
        mean = np.mean(x_train)
        std = np.std(x_train)
        x_train = (x_train-std)/mean
        x_test = (x_test-std)/mean
        x_train = np.expand_dims(x_train, -1)
        x_test = np.expand_dims(x_test, -1)

    return (x_train, x_test)

def reshape_dataset(x_train, y_train, x_test, y_test):
    """Reshape Computer Vision and Speech datasets."""

    num_pixels = x_test.shape[1]*x_test.shape[2]

    # TODO: Change Reshape Function to the format (num_samples, x_axis, y_axis, channels)
    #x_test = x_test.reshape(x_test.shape[0], num_pixels).astype('float32')
    #x_train = x_train.reshape(x_train.shape[0], num_pixels).astype('float32')

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    return (x_train, y_train), (x_test, y_test)


def main():

    # Hyperparameters
    # layers = 1
    # layer_units = 1000
    # Final hyperparameters
    epochs = 100
    batch_size = 50
    # learning_rate = [1, 0.1, 0.001, 0.0001, 0.00001]
    learning_rate = [0.001]
    #transformation_types : "rotate" or "zoom" or "translate" or "none" or "all"
    transformation_type = "all"
    # nr of times the dataset is duplicated
    duplicate_dataset_x_times = 1

    # Dataset : "computer_vision" or "speech_recognition"
    dataset = "computer_vision"
    # Import Datasets

    # printing basic info about the dataset
    # Model / data parameters
    (x_train, y_train), (x_test, y_test) = choose_dataset(dataset, transformation_type)
    for _ in duplicate_dataset_x_times:
      (x_train2, y_train2), (x_test2, y_test2) = choose_dataset(dataset, transformation_type)
      x_train = np.concatenate((x_train,x_train2), axis=0)
      y_train = np.concatenate((y_train,y_train2), axis=0)
      x_test = np.concatenate((x_test,x_test2), axis=0)
      y_test = np.concatenate((y_test,y_test2), axis=0)

    print("dataset with load data:", dataset)
    print("x_train shape:", x_train.shape)
    print("y_train shape:", y_train.shape)
    print(x_train.shape[0], "train samples")
    print(x_test.shape[0], "test samples")

    # Generate and train model
    # TODO: Change inputs to generic_vns_function
    # model = generic_vns_function(x_train.shape[1], layers, y_train.shape[1], layer_units, lr)

    nr_of_classes = y_train.shape[1]
    # Loop over several learning rates
    for lr in learning_rate:
        # Generate and train model
        model = generic_vns_function((x_train.shape[1], x_train.shape[2],
                                    x_train.shape[3]), layers, nr_of_classes, lr)
        # def generic_vns_function(input_dim, number_dense_layers, classes, units, lr):
        trained_model = train_model(model, epochs, batch_size, x_train,
                                    y_train, x_test, y_test)

        # Save model to h5 file
        # trained_model.save('models/model_%s_a4.h5' % dataset)

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

def move_array(X, transform_range=60):
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
