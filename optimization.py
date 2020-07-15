import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np
import matplotlib.pyplot as plt

BATCH_SIZE = 50
EPOCHS = 5

################
# DO NOT MODIFY
NUM_CLASSES = 18
IMG_HEIGHT = 100
IMG_WIDTH = 100
TRAIN_SIZE = 545
TEST_SIZE = 201
################

# this function converts the test and train images into tensors
# the inputs are the jpg files, the labels are the directory names
# returns the generators for the train and test data
def preprocess():
    # Generator for our training data, normalizes the data
    train_image_generator = ImageDataGenerator(rescale=1./255)
    # Generator for our validation data, normalizes the data
    validation_image_generator = ImageDataGenerator(rescale=1./255)

    # assign variables with the proper file path for train and test set
    train_dir = os.path.join(os.getcwd(), 'train')
    test_dir = os.path.join(os.getcwd(), 'test')

    # convert all the images in a directory into a format that tensorflow
    # can work with
    train_data_gen = train_image_generator.flow_from_directory(
                    batch_size=BATCH_SIZE,
                    directory=train_dir,
                    shuffle=True,
                    target_size=(IMG_HEIGHT, IMG_WIDTH),
                    class_mode='categorical')
    test_data_gen = validation_image_generator.flow_from_directory(
                    batch_size=BATCH_SIZE,
                    directory=test_dir,
                    target_size=(IMG_HEIGHT, IMG_WIDTH),
                    class_mode='categorical')
    # return the two objects containing our formatted train and test data
    return (train_data_gen, test_data_gen)

# creates and compiles the model
def train(train_data_gen, test_data_gen):
    ####################################################################
    # Modify the layers here. You can change the number and order of layers,
    # and the hyperparameters of each layer. For example. The Conv2D layer has
    # filters, which represent the dimensionality of the output space, and
    # kernel_size, which represents height and width of the 2D convolution
    # window. Refer to the tensorflow documentation on layers for more details
    # on which hyperparameters you are able to tune.
    # https://www.tensorflow.org/api_docs/python/tf/keras/layers/
    # create the model
    model = tf.keras.Sequential([
        keras.layers.Conv2D(filters=4, kernel_size=3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    ####################################################################
    # Prints a string summary of the network
    model.summary()
    # compile the model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # fit and test the model. Collect statistics after every epoch
    history = model.fit(x=train_data_gen,
                        steps_per_epoch=TRAIN_SIZE// BATCH_SIZE,
                        epochs=EPOCHS,
                        validation_data=test_data_gen,
                        validation_steps=TEST_SIZE  // BATCH_SIZE)
    return (history, model)

# this function is for visualizing accuracy and loss after each epoch
def show_results(history):
    # get array of accuracy values after each epoch for training and testing
    train_acc = history.history['accuracy']
    test_acc = history.history['val_accuracy']

    # get array of loss values after each epoch for training and testing
    train_loss=history.history['loss']
    test_loss=history.history['val_loss']

    print("Final Train Accuracy:", train_acc[-1])
    print("Final Test Accuracy:", test_acc[-1])

    # generate an array for they x axis values
    epochs_range = range(EPOCHS)

    # plot accuracy
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_acc, label='Train Accuracy')
    plt.plot(epochs_range, test_acc, label='Test Accuracy')
    plt.legend(loc='lower right')
    plt.title('Train and Test Accuracy')
    # plot loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_loss, label='Train Loss')
    plt.plot(epochs_range, test_loss, label='Test Loss')
    plt.legend(loc='upper right')
    plt.title('Train and Test Loss')
    plt.show()

(train_data_gen, test_data_gen) = preprocess()
(history, model) = train(train_data_gen, test_data_gen)
show_results(history)
