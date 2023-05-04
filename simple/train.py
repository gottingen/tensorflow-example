import keras
import tensorflow as tf
import numpy as np

class LinerModel(tf.keras.Model):
    def __init__(self):
        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.Dense(units=1, input_dim=1))
        self.model.compile(optimizer='sgd', loss='mse')

    def call(self, inputs):
