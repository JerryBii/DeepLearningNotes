"""
This is a starter file to get you going. You may also include other files if you feel it's necessary.

Make sure to follow the code convention described here:
https://github.com/UWARG/computer-vision-python/blob/main/README.md#naming-and-typing-conventions

Hints:
* The internet is your friend! Don't be afraid to search for tutorials/intros/etc.
* We suggest using a convolutional neural network.
* TensorFlow Keras has the CIFAR-10 dataset as a module, so you don't need to manually download and unpack it.
"""

# Import whatever libraries/modules you need

import numpy as np

# Your working code here

import tensorflow as tf
import matplotlib.pyplot as plt

print(tf.__version__)

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

# A sequential model is what you're going to use most of the time. It just means things are going to go in direct
# order. A feed forward model. No going backwards...for now.
model = tf.keras.models.Sequential()

# need to take this 28x28 image, and make it a flat 1x784
model.add(tf.keras.layers.Flatten())

# These are the layers two being used here, Dense: "fully connected," where each node connects to each prior and
# subsequent node
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))

# Activation function 10 because dataset is numbers 0-9
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

# train model
model.compile(optimizer='adam',  # default optimizer
              loss="sparse_categorical_crossentropy",
              # how will we calculate our "error." Neural network aims to minimize loss.
              metrics=["accuracy"])

model.fit(x_train, y_train, epochs=3)

# Getting a high accuracy and low loss might mean your model learned how to classify digits in general (it
# generalized)...or it simply memorized every single example you showed it (it overfit). This is why we need to test
# on out-of-sample data (data we didn't use to train the model).
val_loss, val_acc = model.evaluate(x_test, y_test)
print(val_loss, val_acc)

model.save("epic_model")
new_model = tf.keras.models.load_model("epic_model")

# probability distributions
predictions = new_model.predict([x_test])
print(np.argmax(predictions[0]))

plt.imshow(x_test[0])
plt.show()

# print(x_train[0])
#
# plt.imshow(x_train[0], cmap=plt.cm.binary)
# plt.show()
# print(x_train[0])
