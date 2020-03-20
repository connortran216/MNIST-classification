# core and utility packages
import pandas as pd
import numpy as np
import itertools
import os
import sys


# visualization
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
#%matplotlib inline
sns.set(style='white', context='notebook', palette='deep')

# modeling
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# keras
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
from keras import models

np.random.seed(2)

# Load the data
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

print(train.head())

Y_train = train["label"]

# Drop 'label' column
X_train = train.drop(labels = ["label"],axis = 1) 

# free some space
del train 

g = sns.countplot(Y_train)

# Normalize the data
X_train = X_train / 255.0
test = test / 255.0

print(X_train.shape)
print(test.shape)

# Reshape image in 3 dimensions (height = 28px, width = 28px , canal = 1), the first dimension of X_train/test is samples
X_train = X_train.values.reshape(-1,28,28,1)
test = test.values.reshape(-1,28,28,1)
print(X_train.shape)

# Encode labels to one hot vectors (ex : 2 -> [0,0,1,0,0,0,0,0,0,0])
Y_train_value = Y_train # keep the origianl label
Y_train = to_categorical(Y_train, num_classes = 10)
print(Y_train.shape)
#Y_train_value = np.argmax(Y_train, axis=1) # keep the origianl label

# Set the random seed
random_seed = 2

# Split the train and the validation set for the fitting
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state=random_seed)
print(X_train.shape)

# preview the images first
plt.figure(figsize=(12,4.8))
x, y = 10, 4
for i in range(40):  
    plt.subplot(y, x, i+1)
    plt.imshow(X_train[i].reshape((28,28)),interpolation='nearest')
    plt.axis('off')
plt.subplots_adjust(wspace=0.1, hspace=0.1)
#plt.show()

#============================DEFINE MODEL CNN=======================================
#===================================================================================
model = Sequential()

model.add(Conv2D(filters = 16, kernel_size = (3, 3), activation='relu', input_shape = (28, 28, 1)))

model.add(BatchNormalization())
model.add(Conv2D(filters = 16, kernel_size = (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(strides=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters = 32, kernel_size = (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(filters = 32, kernel_size = (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(strides=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
model.summary()


# Define the optimizer
optimizer = Adam(lr=1e-4)

# Compile the model
model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])

# With data augmentation to prevent overfitting

datagen = ImageDataGenerator(
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1)  # randomly shift images vertically (fraction of total height)

datagen.fit(X_train)

annealer = LearningRateScheduler(lambda x: 1e-3 * 0.9 ** x)

# Turn epochs to 30 to get 0.9967 accuracy
epochs = 30 
batch_size = 64


# Fit the model
history = model.fit_generator(datagen.flow(X_train,Y_train, batch_size=batch_size),
                              epochs = epochs, 
                              validation_data = (X_val,Y_val),
                              workers = 4,
                              verbose = 2, steps_per_epoch=X_train.shape[0] // batch_size
                              , callbacks=[annealer])
                              



# Score trained model.
scores = model.evaluate(X_val, Y_val, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

# Save model and weights
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'mnist_trained_model_c1.h5'
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)

# Plot the loss and accuracy curves for training and validation 
fig, ax = plt.subplots(2,1)
#ax[0].plot(history.history['loss'], color='b', label="Training loss")
ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])
legend = ax[0].legend(loc='best', shadow=True)

#ax[1].plot(history.history['acc'], color='b', label="Training accuracy")
ax[1].plot(history.history['val_accuracy'], color='r',label="Validation accuracy")
legend = ax[1].legend(loc='best', shadow=True)

#------------------TEST PLOT IMAGE------------------------------------------------------
mnist_classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
y_pred_test = model.predict_proba(X_val)
y_pred_test_classes = np.argmax(y_pred_test, axis=1)
y_pred_test_max_probas = np.max(y_pred_test, axis=1)

cols = 8
rows = 2


fig = plt.figure(figsize=(2 * cols - 1, 3 * rows - 1))
for i in range(cols):
    for j in range(rows):
        random_index = np.random.randint(0, len(Y_val))
        ax = fig.add_subplot(rows, cols, i * rows + j + 1)
        ax.grid('off')
        ax.axis('off')
        ax.imshow(X_val[random_index, :].reshape((28,28)),interpolation='nearest')
        pred_label =  mnist_classes[y_pred_test_classes[random_index]]
        pred_proba = y_pred_test_max_probas[random_index]
        true_label = mnist_classes[Y_val[random_index, 0]]
        ax.set_title("pred: {}\nscore: {:.3}\ntrue: {}".format(
               pred_label, pred_proba, true_label
        ))


plt.show()

# # preview the images first
# plt.figure(figsize=(12,4.8))
# x, y = 10, 4
# for i in range(40):  
    # plt.subplot(y, x, i+1)
    # plt.imshow(X_train[i].reshape((28,28)),interpolation='nearest')
    # plt.axis('off')
# plt.subplots_adjust(wspace=0.1, hspace=0.1)
# #plt.show()




