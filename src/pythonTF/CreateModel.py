"""
CreateModel will create simple TF models for testing on CUDA
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os


class SimpleTFModel(object):

    optimizer = 'adam'

    def __init__(self, name, save_path='./'):
        self.save_path = save_path
        self.name = name
        self.full_path = os.path.join(self.save_path, self.name)

        # create a placeholder for the model
        self.model = None

    def create_model(self):
        assert NotImplemented('You must implement the create_model function')

    @staticmethod
    def loss():
        assert NotImplemented('You must implement the loss function')

    def train_model(self, dataset):
        # setup the model training
        self.model.compile(optimizer=self.optimizer,
                           loss=self.loss(),
                           metrics=['accuracy'])

        # now train the model
        history = self.model.fit(
            dataset.x_train,
            dataset.y_train,
            epochs=10
        )

        y = self.model.predict(dataset.x_train)
        ac = np.abs((dataset.y_train - y[:, 0]) / dataset.y_train)
        plt.scatter(np.diff(dataset.x_train, axis=1), y)
        plt.show()
        a = 5

    def save_model(self):
        self.model.save(self.full_path)


class ANN(SimpleTFModel):

    hidden_layers = [128, 128]
    num_inputs = 2
    num_outputs = 1

    def create_model(self):
        model = []

        # add the first reshape layer
        # model.append(tf.keras.layers.Reshape(t))

        # add the hidden layers
        for hidden_layer in self.hidden_layers:
            model.append(tf.keras.layers.Dense(units=hidden_layer, activation='relu'))

        # add the output layer
        model.append(tf.keras.layers.Dense(units=self.num_outputs))

        # now create the model
        self.model = tf.keras.Sequential(model)

    @staticmethod
    def loss():
        return tf.keras.losses.MeanSquaredError(reduction="auto", name="mean_squared_error")

