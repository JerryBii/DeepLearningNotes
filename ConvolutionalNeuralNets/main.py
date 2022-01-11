import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D

# High level api to visualize model learning
from tensorflow.keras.callbacks import TensorBoard
import pickle
import numpy as np
import time

NAME = "Cats-vs-dog-cnn-64x2-{}".format(int(time.time()))

tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))

gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.333)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

X = pickle.load(open("X.pickle", "rb"))
y = pickle.load(open("y.pickle", "rb"))
y = np.array(y)

# normalize
# image data min: 0 max: 255
# therefore, divide by 255
X = X / 255.0

model = Sequential()

# First layer
model.add(Conv2D(64,  # Convolution (2D here requires a 2D dataset)
          (3, 3),
          input_shape=X.shape[1:]  # X is numpy array where X[0] is always -1 i.e. we don't care
          ))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Conv2D(64,  # 2D convolution
          (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(64))  # Dense layers require 1D dataset
model.add(Activation("relu"))

# output layer
model.add(Dense(1))
model.add(Activation("sigmoid"))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


model.fit(X, y, batch_size=12, epochs=22, validation_split=0.1, callbacks=[tensorboard])


