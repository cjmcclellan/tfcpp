import numpy as np


class Test(object):

    def __int__(self):
        self.data = None

    def receive_data(self, data):
        self.data = np.array(data)

    def process_data(self):
        self.data += 1

    def return_data(self):
        return self.data


def runTest(data):
    # print(x)
    # print("hello")
    t = Test()
    t.receive_data(data)
    t.process_data()
    d = t.return_data()
    print(d.shape)
    print(d[:10])