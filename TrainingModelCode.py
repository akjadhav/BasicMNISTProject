from google.colab import drive
drive.mount('/content/drive')

!pip install tensorflow==2.0.0-beta0

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

import keras
from keras import layers
from keras import Model
from keras import models

from sklearn.metrics import classification_report, confusion_matrix

import matplotlib.pyplot as plt
from keras.utils import np_utils
# %matplotlib inline

mnist = tf.keras.datasets.mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()

plt.imshow(X_train[0], cmap='gray')

fig = plt.figure()
for i in range(9):
  plt.subplot(3,3,i+1)
  plt.tight_layout()
  plt.imshow(X_train[i], cmap='gray', interpolation='none')
  plt.title("Digit: {}".format(y_train[i]))
  plt.xticks([])
  plt.yticks([])

X_train = X_train.reshape(60000, 28, 28, 1) / X_train.max()
X_test = X_test.reshape(10000, 28, 28, 1) / X_test.max()

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.summary()

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=5)

test_scores = model.evaluate(X_test, y_test)
print('Test loss:', test_scores[0])
print('Test accuracy:', test_scores[1])

predictions = model.predict_classes(X_test)

print('Classification Report: ')
print(classification_report(y_test, predictions))
print('\n')
print('Confusion Matrix: ')
cm = confusion_matrix(y_test, predictions)
print(cm)

model.save('BasicMNISTModel.h5')
