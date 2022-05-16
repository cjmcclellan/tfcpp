import tensorflow as tf
import numpy as np
from tensorflow import keras
import os
import time
import h5py


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
    B = N
    t_in = keras.Input(shape=(N,), dtype=dtype, name="input")
    temperature_response_mat = ConstantLayer(mat, dtype=dtype, name="matrix")(t_in)
    t_output_samples = tf.linalg.matmul(t_in, tf.linalg.matrix_transpose(temperature_response_mat))

    return keras.Model(
        inputs=[t_in],
        outputs=[temperature_response_mat, t_output_samples],
    )

def my_func(arg, dtype):
    arg = tf.convert_to_tensor(arg, dtype=dtype)
    return arg


def saveH5(N, D, dtype=tf.double):
    mat = np.random.random((N, D))
    h5f = h5py.File("./testMat.h5", "w")
    h5f.create_dataset('dataset_1', data=mat)
    h5f.close()


def buildBigModel(N, D, dtype=tf.double):

    mat = np.random.random((N, D))

    # tf.compat.v1.disable_eager_execution()
    t_in = keras.Input(shape=(D,), dtype=dtype, name="input")
    # t_in = tf.compat.v1.placeholder(dtype, name='input', shape=(D, None))
    temperature_response_mat = tf.constant(mat, dtype)
    # temperature_response_mat = ConstantLayer(mat, dtype=dtype, name="matrix")(t_in)
    t_output_samples = tf.linalg.matmul(t_in, tf.linalg.matrix_transpose(temperature_response_mat), name='output')
    # sess = tf.compat.v1.Session()
    # tf.io.write_graph(sess.graph_def, './tmp/modelgraph', 'model.pbtxt')
    return keras.Model(
        inputs=[t_in],
        outputs=[t_output_samples],
    )


def saveModel(model, save_path):
    tf_save_path = os.path.join(save_path, 'tfmodel.h5')
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    model.save(tf_save_path)
    # tf.keras.models.save_model(model, tf_save_path, include_optimizer=False, save_format="tf")


def runModel(path, D):
    start = time.time()
    model = tf.keras.models.load_model(os.path.join(path, 'tfmodel'))
    a = model(np.ones((1, D)))
    end = time.time()
    print('### loading took {0} seconds'.format(end - start))
    start = time.time()
    model = tf.keras.models.load_model(os.path.join(path, 'tfmodel'))
    a = model(np.ones((1, D)))
    end = time.time()
    print('### second run took {0} seconds'.format(end - start))



if __name__ == '__main__':
    # modelSizes = [100, 1000]
    # for N in modelSizes:
    #     N = int(N)
    #
    #     # create and save the double
    #     model = buildModel(N, tf.double)
    #     saveModel(model, './testDModel_N={0}'.format(N))
    #
    #     # create and save the float
    #     model = buildModel(N, tf.float32)
    #     saveModel(model, './testFModel_N={0}'.format(N))

    # create and save the double
    D = 600
    N = 50000
    saveH5(N, D)
    # model = buildBigModel(N, D, tf.double)
    path = './testDModel_D={0}_N={1}'.format(N, D)
    # saveModel(model, path)
    print('############### model saved ################')
    # runModel(path, D)

