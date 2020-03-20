from __future__ import print_function
import keras
from keras.datasets import cifar10

# core and utility packages
import pandas as pd
import numpy as np
import itertools
import os
import sys

import tensorlayer as tl
import numpy as np
# visualization
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns

# modeling
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# keras
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding

#==========================================================================================

# Load the data
import tensorflow as tf
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Reshaping the array to 4-dims so that it can work with the Keras API
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)

x_test_ori = x_test
y_test_ori = y_test
# Making sure that the values are float so that we can get decimal points after division
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
# Normalizing the RGB codes by dividing it to the max RGB value.
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print('Number of images in x_train', x_train.shape[0])
print('Number of images in x_test', x_test.shape[0])

mnist_classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)
print("Datatype:" ,y_test.shape)
# Split the train and the validation set for the fitting
# Set the random seed
random_seed = 2
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size = 0.1, random_state=random_seed)

# # preview the images first
# plt.figure(figsize=(12,4.8))
# x, y = 10, 4
# for i in range(40):  
    # plt.subplot(y, x, i+1)
    # plt.imshow(x_train[i].reshape((28,28)),interpolation='nearest')
    # plt.axis('off')
# plt.subplots_adjust(wspace=0.1, hspace=0.1)
# plt.show()




# #------------------Load Weight---------------------
#----------------LOAD MODEL--------------------------
from keras.models import load_model
 
# load model
model = load_model('mnist_trained_model_c1.h5')
# summarize model.
model.summary()
# evaluate the model
score = model.evaluate(x_test, y_test, verbose=1)
print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))

#------------------TEST PLOT IMAGE------------------------------------------------------
#from sklearn.preprocessing import OneHotEncoder

y_pred_test = model.predict_proba(x_test)
y_pred_test_classes = np.argmax(y_pred_test, axis=1)
y_pred_test_max_probas = np.max(y_pred_test, axis=1)

cols = 8
rows = 2


#cifar10_classes = OneHotEncoder(cifar10_classes)
fig = plt.figure(figsize=(2 * cols - 1, 3 * rows - 1))
for i in range(cols):
    for j in range(rows):
        random_index = np.random.randint(0, 6000)
        ax = fig.add_subplot(rows, cols, i * rows + j + 1)
        ax.grid('off')
        ax.axis('off')
        ax.imshow(x_test[random_index, :].reshape((28,28)),interpolation='nearest')
        pred_label =  mnist_classes[y_pred_test_classes[random_index]]
        pred_proba = y_pred_test_max_probas[random_index]
        #true_label = mnist_classes[y_test_ori[random_index, 0]]
        ax.set_title("pred: {}\nscore: {:.3}".format(
               pred_label, pred_proba#, true_label
        ))
plt.show()