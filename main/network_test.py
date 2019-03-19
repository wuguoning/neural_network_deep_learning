import sys
import os

module_path = os.path.abspath(os.path.join('../src/mnist_loader'))
if module_path not in sys.path:
    sys.path.append(module_path)

import mnist_loader

module_path = os.path.abspath(os.path.join('../src/network'))
if module_path not in sys.path:
    sys.path.append(module_path)

import network

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
net = network.Network([784, 100, 30, 10])
net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
