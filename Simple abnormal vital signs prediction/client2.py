import flwr as fl
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import tensorflow as tf
import seaborn as sns
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers,Sequential
from sklearn.metrics import confusion_matrix


# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
df = pd.read_csv('newdata.csv')
X = df.iloc[:, 1:-1].values #[:1000000]
y = df.iloc[:, -1].values #[:1000000]
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

y_test = tf.one_hot(tf.cast(tf.reshape(y_test, -1), dtype=tf.int32), depth=4)
y_train = tf.one_hot(tf.cast(tf.reshape(y_train, -1), dtype=tf.int32), depth=4)

model = Sequential()

model.add(layers.Dense(128, input_shape=[6]))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(4, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# model = tf.keras.applications.MobileNetV2((32,32,3), classes=10, weights=None)
# model.compile('adam', 'sparse_categorical_crossentropy', metrics=['accuracy'])

class CifarClient(fl.client.NumPyClient):
    def get_parameters(self):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        model.fit(x_train, y_train, epochs=15, batch_size=32, steps_per_epoch=3)
        return model.get_weights(), len(x_train), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(x_test, y_test)
        return loss, len(x_test), {"accuracy": accuracy}

fl.client.start_numpy_client(server_address="localhost:8080", client=CifarClient())