from google.colab import drive
drive.mount('/content/drive')

"""# Installing TensorFlow 2.0 **Beta**"""

!pip install tensorflow==2.0.0-beta0

# import statements to import all the libraries that we use

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

import keras
from tensorflow.python.keras import layers
from tensorflow.python.keras import Model
from tensorflow.python.keras import models

from sklearn.metrics import classification_report, confusion_matrix

import matplotlib.pyplot as plt
from keras.utils import np_utils
# %matplotlib inline

mnist = tf.keras.datasets.mnist # importing the MNIST dataset from Keras

(X_train, y_train), (X_test, y_test) = mnist.load_data() # Loading the MNIST data into local variables

fig = plt.imshow(X_train[0], cmap='gray') # Shows an example image of the MNIST dataset 
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)

# Gives the first 9 images from the dataset as example images

fig = plt.figure()
for i in range(9):
  plt.subplot(3,3,i+1)
  plt.tight_layout()
  plt.imshow(X_train[i], cmap='gray', interpolation='none')
  plt.title("Digit: {}".format(y_train[i]))
  plt.xticks([])
  plt.yticks([])

X_train = X_train.reshape(60000, 28, 28, 1) / X_train.max() # Normalizes the training data and reshapes to the needed dimensions
X_test = X_test.reshape(10000, 28, 28, 1) / X_test.max()  # Normalizes the testing data and reshapes to the needed dimensions

"""# Defining **the** Layers in the Model"""

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

"""# **This block is for TensorBoard only**"""

!pip install -q tf-nightly-2.0-preview

# Load the TensorBoard notebook extension
# %load_ext tensorboard

import tensorflow as tf
import datetime

!rm -rf ./logs/ 

log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

"""# **Training the Model**"""

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=5, callbacks=[tensorboard_callback])

"""# **Metrics**"""

# Accuracy vs. Loss Graph from the Model Training

plt.figure(figsize = (8,5))
plt.plot(history.history['accuracy'], color = 'blue')
plt.plot(history.history['loss'], color = 'red')
plt.title('Accuracy vs. Loss')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.legend(['Accuracy', 'Loss'])

"""# **Model Accuracy**"""

# Tests the Model on the Test Data to determine an Accuracy

test_scores = model.evaluate(X_test, y_test)
print('Test loss:', test_scores[0])
print('Test accuracy:', test_scores[1])

"""# **Confusion Matrix and Classification Report**"""

# Creates Predictions on the Test Data

predictions = model.predict_classes(X_test)

# Classification Report and Confusion Matrix for the Model

print('Classification Report: ')
print(classification_report(y_test, predictions))
print('\n')
print('Confusion Matrix: ')
cm = confusion_matrix(y_test, predictions)
print(cm)

"""# **Saving the Model**"""

model.save('BasicMNISTModel.h5')

"""# Model Visualization Using **TensorBoard**"""

# %tensorboard --logdir logs/fit
