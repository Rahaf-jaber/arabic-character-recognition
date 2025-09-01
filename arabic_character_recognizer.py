#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cleaned script auto-generated from Arabic_Character_Recognizer_complex_CNN.ipynb.
 - Markdown headings are preserved as comments.
 - IPython magics and shell commands are removed.
 - Cells are concatenated in notebook order.
"""

from __future__ import annotations
# #Arabic Character Recognition

# ## Data Exploring & Preprocessing

# ---- Code cell 1 ----
from google.colab import drive
drive.mount('/content/drive')

# ---- Code cell 2 ----
import tensorflow as tf                                                         # The main framework we will build our model with.
import numpy as np                                                              # Used for mathimatical operations.
import pandas as pd                                                             # Will be used to load our data frame.
import cv2                                                                      # Used for image processing.
from matplotlib import pyplot as plt                                            # Used for plottin our data.
from tensorflow.keras.utils import to_categorical                               # Utility in Tensorflow to convert our true category values.

# ---- Code cell 3 ----
path = '/content/drive/MyDrive/archive (3)'              # Here we specify the path to our data location on my drive
train_data_x = pd.read_csv(path + '/csvTrainImages 13440x1024.csv', header=None)# Then we load the training images.
train_data_y = pd.read_csv(path + '/csvTrainLabel 13440x1.csv', header=None)    # Training labels.
test_data_x = pd.read_csv(path + '/csvTestImages 3360x1024.csv', header=None)   # Testing images.
test_data_y = pd.read_csv(path + '/csvTestLabel 3360x1.csv', header=None)       # Testing labels.

# ---- Code cell 4 ----
print('We have  %d training images each contains %d pixels.' %(train_data_x.shape[0], train_data_x.shape[1]))
print('We have  %d training labels each contains %d classes.' %(train_data_y.shape[0], len(train_data_y.value_counts())))
print('We have  %d testing images each contains %d pixels.' %(test_data_x.shape[0], test_data_x.shape[1]))
print('We have  %d testing labels each contains %d classes.' %(test_data_y.shape[0], len(test_data_y.value_counts())))

# ---- Code cell 5 ----
train_data_y.value_counts()

# ---- Code cell 6 ----
fig = plt.figure(figsize=(8, 8))                                                # Setting the figure size.
columns = 4                                                                     # Selecting the number of columns.
rows = 5                                                                        # Selectin the number of rows.
for i in range(1, columns*rows +1):                                             # Looping through rows & columns.
  img = test_data_x.iloc[i].to_numpy().reshape((32,32))                         # Reshaping the image into its size 32x32
  fig.add_subplot(rows, columns, i)                                             # Adding the image to the plot
  plt.imshow(img, cmap='gray')                                                  # Showing the image using plt
plt.show()                                                                      # Finally shpwing the whole plot containing all the subplots

# ---- Code cell 7 ----
def preprocess_data(train_data_x):
  train_data_x = train_data_x.to_numpy().reshape((train_data_x.shape[0], 32, 32)).astype('uint8')
  for i in range(len(train_data_x)):
    train_data_x[i] = cv2.rotate(train_data_x[i], cv2.ROTATE_90_CLOCKWISE)      # Rotating the images.
    train_data_x[i] = np.flip(train_data_x[i], 1)                               # Flipping the images
  train_data_x = train_data_x.reshape([-1, 32, 32, 1]).astype('uint8')          # Reshaping into the required size.
  train_data_x = train_data_x.astype('float32')/255                             # Here we normalize our images.
  return np.asarray(train_data_x)

# ---- Code cell 8 ----
train_x = preprocess_data(train_data_x)                                         # Returns an array of dimensions (13440,32,32,1).
test_x = preprocess_data(test_data_x)                                           # Returns an array of dimensions (3360,32,32,1).

# ---- Code cell 9 ----
train_y = to_categorical(train_data_y.values.astype('int32') - 1                # Returns an array of dimentions (13340, 28).
                         , num_classes=28)
test_y = to_categorical(test_data_y.values.astype('int32') - 1                  # Returns an array of dimentions (3360, 28).
                        , num_classes=28)

# ---- Code cell 10 ----
from sklearn.utils import shuffle                                               # Importing shuffle function from sklearn library.
train_x, train_y = shuffle(train_x, train_y)                                    # Now we shuffle x & y in the training set.
test_x, test_y, shuffle(test_x, test_y)                                         # Then x & y in our testing set.

# ## Building Model

# ---- Code cell 11 ----
def create_model(activation='relu', optimizer='adam', kernel_initializer='he_normal'):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), padding='same', input_shape=(32, 32, 1), activation=activation, kernel_initializer=kernel_initializer),
        tf.keras.layers.MaxPool2D(2, 2),
        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation=activation, kernel_initializer=kernel_initializer),
        tf.keras.layers.MaxPool2D(2, 2),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation=activation, kernel_initializer=kernel_initializer),
        tf.keras.layers.MaxPool2D(2, 2),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Conv2D(256, (3, 3), padding='same', activation=activation, kernel_initializer=kernel_initializer),
        tf.keras.layers.MaxPool2D(2, 2),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(256, activation=activation, kernel_initializer=kernel_initializer, kernel_regularizer='l2'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),

        tf.keras.layers.Dense(128, activation=activation, kernel_initializer=kernel_initializer, kernel_regularizer='l2'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),

        tf.keras.layers.Dense(28, activation='softmax', kernel_initializer=kernel_initializer)
    ])

    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# ---- Code cell 12 ----
model = create_model()                                                          # Now we created an instance of a model with our custom architefture.
model.summary()                                                                 # Then we display our model's summary.

# ---- Code cell 13 ----
model = create_model(optimizer='RMSprop',                                       # We create our model with the specified hyper parameters
                     kernel_initializer='normal',
                     activation='relu')

# ---- Code cell 14 ----
from keras.callbacks import ModelCheckpoint                                     # We will import a call back to save the best epoch's weights

checkpointer = ModelCheckpoint(filepath='weights.hdf5', verbose=1, save_best_only=True)
history = model.fit(train_x,
                    train_y,
                    validation_split= 0.3,                                      # The model will split the data into 30% of validation.
                    epochs=30,                                                  # We will run the model for 30 epochs
                    batch_size=64,                                              # We will have a batch size of 64
                    callbacks=[checkpointer])                                   # Finally we will use the imported callback

# ---- Code cell 15 ----
model.load_weights('weights.hdf5')                                              # Loading the best weights
model.evaluate(test_x, test_y)                                                  # Evaluating our model

# ---- Code cell 16 ----
# PLOT LOSS AND ACCURACY

import matplotlib.image  as mpimg
import matplotlib.pyplot as plt

#-----------------------------------------------------------
# Retrieve a list of list results on training and test data
# sets for each training epoch
#-----------------------------------------------------------
acc=history.history['accuracy']
val_acc=history.history['val_accuracy']
loss=history.history['loss']
val_loss=history.history['val_loss']

epochs=range(len(acc)) # Get number of epochs

#------------------------------------------------
# Plot training and validation accuracy per epoch
#------------------------------------------------
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Training and validation accuracy')
plt.legend(['train', 'val'], loc='upper left')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.figure()

#------------------------------------------------
# Plot training and validation loss per epoch
#------------------------------------------------
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Training and validation loss')
plt.legend(['train', 'val'], loc='upper left')
plt.ylabel('accuracy')
plt.xlabel('epoch')

plt.title('Training and validation loss')

# ---- Code cell 17 ----
model.save('/content/gdrive/MyDrive/Arabic_OCR2.h5')
