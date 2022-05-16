
import os
import subprocess

"""
This python script will run a series of TFvsCublas tests at different batch numbers, batch sizes, and input/output size.
If possible, it will also call nsys
"""


def main():

    nsys_path = "/opt/nvidia/nsight-systems/2022.2.1/bin/nsys profile -t cuda,cublas --gpu-metrics-device=0 "

    binary_path = "../build/test_tfvscublas"

    # numBatches = [10, 100, 1000]
    # batchSizes = [100, 1000, 2000]
    # modelSizes = [40, 400, 4000, 16000]

    numBatches = [10, 100]
    batchSizes = [100, 1000]
    modelSizes = [100, 1000]

    outputs = []
    with open('./RunTFvsCublas.log', 'w+') as f:

        for numB in numBatches:
            for bSize in batchSizes:
                for mSize in modelSizes:
                    outfile = "nsys_profile_numBatch={0}_batchSize={1}_modelSize={2}".format(numB, bSize, mSize)
                    command = "{4} --output={5} {0} --numB {1} --bSize {2} --modelSize {3}".format(binary_path, numB, bSize, mSize, nsys_path, outfile)

                    p = subprocess.Popen(command.split(' '), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    stdout, stderr = p.communicate()
                    status = "\n\n############  numBatch={0}  batchSize={1}  modelSize={2}  #############\n\n".format(numB, bSize, mSize)
                    print(status)
                    print(stdout.decode("utf-8"))
                    f.write(status)
                    f.write(stdout.decode("utf-8"))
                    f.write("\n")


if __name__ == '__main__':
    main()

