import tensorflow as tf
import numpy as np
from tensorflow import keras
import os


class ConstantLayer(keras.layers.Layer):
    """Returns a constant value regardless of input
    """
    def __init__(self, constant, dtype="float32", name=None):
        super(ConstantLayer, self).__init__(name=name)
        self.constant = tf.cast(constant, dtype)

    def call(self, inputs):
        return self.constant


def buildModel(N, dtype=tf.double):

    mat = np.random.random((N, N))

    t_in = keras.Input(shape=(N,), dtype=dtype, name="input")
    temperature_response_mat = ConstantLayer(mat, dtype=dtype, name="matrix")(t_in)
    t_output_samples = tf.linalg.matvec(temperature_response_mat, t_in)

    return keras.Model(
        inputs=[t_in],
        outputs=[t_output_samples],
    )


def saveModel(model, save_path):
    tf_save_path = os.path.join(save_path, 'tfmodel')
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    tf.keras.models.save_model(model, tf_save_path, include_optimizer=False, save_format="tf")


if __name__ == '__main__':
    modelSizes = [1e2, 1e3, 1e4]
    for N in modelSizes:
        N = int(N)
        model = buildModel(N)
        saveModel(model, './testModel_N={0}'.format(N))
