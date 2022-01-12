"""
Create the data
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


class DataSet(object):

    def __init__(self):
        self.x_train = []
        self.y_train = []

    def create_data(self):
        # get the data
        inputs, outputs = self._create_data()

        # now save the x and y data
        self.x_train = np.array(inputs)
        self.y_train = np.array(outputs)[:, 0]

        # now wrap that data in a tf keras dataset
        # train_dataset = tf.data.Dataset.from_tensor_slices((inputs, outputs))

        # return train_dataset

    def _create_data(self):
        assert NotImplementedError('You must have a create_data function')
        return [], []

    def norm_data(self, x, a, b):
        return ((x - min(x)) * (b - a)) / (max(x) - min(x)) + a


class ResistorDataSet(DataSet):

    num_examples = 10000

    min_value = -1
    max_value = 1

    resistance = 20

    noise_level = 0.1

    def _create_data(self):
        # create random values
        inputs = np.random.rand(self.num_examples, 2)

        # create random inputs
        inputs = ((inputs - 0) * (self.max_value - self.min_value)) / (1 - 0) + self.min_value

        noise = self.norm_data(np.random.rand(self.num_examples, 1), -self.noise_level, self.noise_level)

        # create the outputs
        outputs = np.diff(inputs, axis=1) / self.resistance

        # add a little noise to the outputs
        outputs = outputs + outputs * noise
        # plt.scatter(np.diff(inputs, axis=1), outputs)
        # plt.show()
        return inputs, outputs
