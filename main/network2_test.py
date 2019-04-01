import sys
import os

# Import the mnist_loader function.
module_path = os.path.abspath(os.path.join('../src/mnist_loader'))
if module_path not in sys.path:
    sys.path.append(module_path)

import mnist_loader

# Import the network function
module_path = os.path.abspath(os.path.join('../src/network'))
if module_path not in sys.path:
    sys.path.append(module_path)

import network2

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
net = network2.Network([784, 100, 30, 10], cost=network2.CrossEntropyCost)
net.large_weight_initializer()
net.SGD(training_data, 30, 10, 0.5, lmbda=5.0, \
        evaluation_data=validation_data,\
        monitor_evaluation_accuracy=True)
