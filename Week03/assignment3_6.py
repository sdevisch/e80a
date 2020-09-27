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
    # model = generic_vns_function((28, 28, 1), layers, (28, 28, 1), layer_units, lr)
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

    #TODO: Add a Convolutional Layer
    #TODO: Add a MaxPool2D layer
    #TODO: Add a Flatten Layer
    # for i in range(number_dense_layers):
        #model.add(layers.Dense(units=units, input_dim=input_dim,
        #          kernel_initializer='normal', activation='relu'))
    #    model.add(layers.Dense(units=units, input_dim=input_dim,
    #              kernel_initializer='normal', activation='relu'))

    #model.add(layers.Dense(classes, kernel_initializer='normal',
    #          activation='softmax'))
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

def choose_dataset(dataset_type):
    """Select dataset based on string variable."""
    if dataset_type == "nlp":
        return get_imdb_dataset(dir=DB_DIR)
    elif dataset_type == "computer_vision":
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    elif dataset_type == "speech_recognition":
        (x_train, y_train), (x_test, y_test), (_, _) = get_speech_dataset()
    else:
        raise ValueError("Couldn't find dataset.")

    (x_train, x_test) = normalize_dataset(dataset_type, x_train, x_test)

    (x_train, y_train), (x_test, y_test) = reshape_dataset(x_train, y_train, x_test, y_test)

    return (x_train, y_train), (x_test, y_test)

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
    layers = 1
    layer_units = 1000
    epochs = 100
    batch_size = 100
    #lr = 0.0001
    #learning_rate = [1, 0.1, 0.001, 0.0001, 0.00001]
    learning_rate = [0.001]

    # Dataset : "computer_vision" or "speech_recognition"
    dataset = "computer_vision"
    # Import Datasets

    # printing basic info about the dataset
    # Model / data parameters
    (x_train, y_train), (x_test, y_test) = choose_dataset(dataset)

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
        model = generic_vns_function((x_train.shape[1], x_train.shape[2], x_train.shape[3]), layers, nr_of_classes, layer_units, lr)
        # def generic_vns_function(input_dim, number_dense_layers, classes, units, lr):
        trained_model = train_model(model, epochs, batch_size, x_train, y_train, x_test, y_test)

        # Save model to h5 file
        trained_model.save('models/model_%s_a3.h5' % dataset)

    return None

if __name__ == '__main__':
    main()
