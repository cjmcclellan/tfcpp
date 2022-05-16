import h5py
import numpy as np


def saveH5(N, D, path, dtype=np.float64):
    mat = np.random.random((N, D)).astype(dtype)
    h5f = h5py.File(path, "w")
    h5f.create_dataset('matrix', data=mat)
    h5f.close()


if __name__ == '__main__':
    D = 600
    N = 50000
    saveH5(N, D, "./d_test.h5", np.float64)
    saveH5(N, D, "./f_test.h5", np.float32)