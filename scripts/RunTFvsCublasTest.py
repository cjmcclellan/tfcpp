import os
import subprocess

"""
This python script will run a series of TFvsCublas tests at different batch numbers, batch sizes, and input/output size.
If possible, it will also call nsys
"""


def main():

    nsys_path = ""

    binary_path = "../build/test_tfvscublas"

    numBatches = [10, 100, 1000]
    batchSizes = [100, 1000, 5000]
    modelSizes = [40, 400, 4000, 16000]

    # numBatches = [10, 100]
    # batchSizes = [100, 1000]
    # modelSizes = [100, 1000]

    outputs = []

    for numB in numBatches:
        for bSize in batchSizes:
            for mSize in modelSizes:
                command = "{0} --numB {1} --bSize {2} --modelSize {3}".format(binary_path, numB, bSize, mSize)

                p = subprocess.Popen(command.split(' '), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                stdout, stderr = p.communicate()
                outputs.append("\n\n############  numBatch={0}  batchSize={1}  modelSize={2}  #############\n".format(numB, bSize, mSize))
                outputs.append(stdout.decode("utf-8"))

    with open('./RunTFvsCublas.log', 'w+') as f:
        f.writelines("\n".join(outputs))


if __name__ == '__main__':
    main()
