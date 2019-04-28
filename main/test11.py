"""network4.py
~~~~~~~~~~~~~~

An improved version of network.py, implementing the stochastic
gradient descent learning algorithm for a feedforward neural network.
Improvements include the addition of the cross-entropy cost function,
Dropout-regularization, and better initialization of network weights.  Note
that I have focused on making the code simple, easily readable, and
easily modifiable.  It is not optimized, and omits many desirable
features.

"""

#### Libraries
# Standard library
import json
import random
import sys

# Third-party libraries
import numpy as np

#### Main Network class
class Network(object):

    def __init__(self, sizes):
        """The list ``sizes`` contains the number of neurons in the respective
        layers of the network.  For example, if the list was [2, 3, 1]
        then it would be a three-layer network, with the first layer
        containing 2 neurons, the second layer 3 neurons, and the
        third layer 1 neuron.  The biases and weights for the network
        are initialized randomly, using
        ``self.default_weight_initializer`` (see docstring for that
        method).

        """
        self.num_layers = len(sizes)
        self.sizes = sizes

    def weights_biases_initializer(self):
        self.weights = [np.random.randn(y,x) for x, y in zip(self.sizes[:-1], self.sizes[1:])]
        self.biases = [np.random.randn(y,1) for y in self.sizes[1:]]

    def dropout_mask(self, prob=0.5):
        """Using binomial distribution to generate a thinner network,
        which is inherit the weights and bias from the mother full
        network. The "prob" is the probability of the binomial
        distribution.

        """
        thin_sizes = [np.ones(self.sizes[0])]
        # To record the selected neurons.
        thin_sizes_num = [self.sizes[0]]
        # To record the numbers of the selected neurons.

        for k in self.sizes[1:-1]:
            binom_dis = np.random.binomial(n=1, p=prob, size=k)
            print(binom_dis)
            thin_sizes_num.append(np.sum(binom_dis))
            thin_sizes.append(binom_dis)
        thin_sizes.append(np.ones(self.sizes[-1]))
        thin_sizes_num.append(self.sizes[-1])

        self.weights_mask = [np.outer(y,x)==1 for x, y in zip(thin_sizes[:-1], thin_sizes[1:])]
        self.biases_mask = [thin_sizes[i]==1 for i in range(1,self.num_layers)]

        return thin_sizes_num

    def thin_weights_biases(self):
        """
        Generate the thin weights and biases using dropout method
        """
        thin_sizes_num = self.dropout_mask(prob=0.5)
        thin_weights = [self.weights[i][self.weights_mask[i]].reshape(thin_sizes_num[i+1],\
                        thin_sizes_num[i]) for i in range(self.num_layers-1)]
        thin_biases = [self.biases[i][self.biases_mask[i]] for i in range(self.num_layers-1)]

        return thin_weights, thin_biases

if  __name__ == '__main__':
    sizes = [2,4,5,2]
    net = Network(sizes)
    net.weights_biases_initializer()
    thin_weights, thin_biases = net.thin_weights_biases()
    print(net.weights)
    print("\n")
    print(thin_weights)
    print("\n")
    print(thin_biases)
    print("\n")



