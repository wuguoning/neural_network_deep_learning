"""
cost_fun.py
A module to implement the cost function test.
Written by wuguonning.
2019-3-19
"""

# Import modules
import numpy as np

class CostFunction(object):

    def __init__(self, weight, bias):
        self.weight = weight
        self.bias = bias

    def GD(self, training_data, output_value, eta):
        """
        Train the neuron using gradient method.
        The "training_data" is a signal data, the eta
        is the learning rate.
        """
        nabla_b = 0
        nabla_w = 0

        # update the weight and bias.
        delta_nabla_w, delta_nabla_b = self.update_gradiet(training_data, output_value)
        nabla_b += delta_nabla_b
        nabla_w += delta_nabla_w
        self.weight = self.weight - eta * nabla_w
        self.bias = self.bias - eta * nabla_b

    def update_gradiet(self, x, y):
        """
        Update the neuron's weight by applying gradient descent
        method. The tuples x is the input and y is the supposed
        output values.
        """
        #neuron_input = w*x +b
        neuron_input = self.weight * x + self.bias
        nabla_w = (sigmoid(neuron_input) - y) * \
            sigmoid_prime(neuron_input) * x
        nabla_b = (sigmoid(neuron_input) - y) * \
            sigmoid_prime(neuron_input)
        return nabla_w, nabla_b


    def CrossEntropy(self, training_data, output_value, eta):
        """
        The Cross-Entropy cost function for this neuron is:
            C = -1/n*(\sum_x [y \ln a + (1-y)\ln (1-a)])
            where n is the total number of items of training data,
            the sum is over all training inputs, x, and y is the
            corresponding desired output.

            It tells us that the rate at which the weight learns
            is controlled by (\sigma(z) - y), i.e., by the error
            in the output. The larger the error, the faster the
            neuron will learn.
        """
        neuron_input = self.weight * training_data + self.bias
        nabla_w = (sigmoid(neuron_input) - output_value) * training_data
        nabla_b = sigmoid(neuron_input) - output_value
        self.weight = self.weight - eta * nabla_w
        self.bias = self.bias - eta * nabla_b


    def evaluate(self, training_data, output_value):
        """
        Evaluate the output of the neuron.
        """
        return sigmoid(self.weight*training_data + self.bias)

# Define the function for output

def sigmoid(x):
    """
    The sigmoid function
    """
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_prime(x):
    """
    Derivative of the sigmoid function
    """
    return sigmoid(x)*(1.0 - sigmoid(x))
