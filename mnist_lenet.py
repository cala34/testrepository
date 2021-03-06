import tensorflow as tf
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import numpy as np
import math

'''
Convolutional Neural Network by LeNet-5 Architecture

This script trains a convoltional neural network with the MNIST data base.
The architecture is according to LeNet-5. It trains the network, prints its
training results and saves the resulting feature vectors to output.npy
'''

def mnist_features():
    # training parameters
    batch_size = 128
    num_classes = 10
    epochs = 3

    # input image dimensions
    img_rows, img_cols = 28, 28

    # load data, split into train and test sets
    (x_train, labels_train), (x_test, labels_test) = mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    # scale the pixel values to [0,1]
    x_train /= 255
    x_test /= 255

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(labels_train, num_classes)
    y_test = keras.utils.to_categorical(labels_test, num_classes)

    # model
    model = Sequential()
    conv1 = Conv2D(6,
                   kernel_size=(5, 5),
                   activation='relu',
                   padding = 'same',
                   input_shape=input_shape)
    model.add(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))
    model.add(pool1)
    conv2 = Conv2D(16, kernel_size=(5, 5), padding = 'valid',
                     activation='relu')
    model.add(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))
    model.add(pool2)
    conv3 = Conv2D(120, kernel_size=(5, 5), padding = 'valid',
                     activation='relu')
    model.add(conv3)
    flat = Flatten()
    model.add(flat)
    model.add(Dense(84, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    # training
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Lenet Test Loss:', score[0])
    print('Lenet Test Accuracy:', score[1])

    # save feature vectors in output
    input = model.input
    outputs = [layer.output for layer in model.layers]
    functor = K.function([input, K.learning_phase()], outputs )
    layer_outs_train = functor([x_train, 1.])
    layer_outs_test = functor([x_test, 1.])
    features_train = layer_outs_train[5]
    features_test = layer_outs_test[5]
    output = [features_train, features_test, y_train, y_test, labels_train, labels_test]

    return output

if __name__ == "__main__":
    print("Saving output to output.npy...")
    np.save("output", mnist_features())
