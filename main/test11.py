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

    def dropout_size(self, prob=0.5):
        """Using binomial distribution to generate a thinner network,
        which is inherit the weights and bias from the mother full
        network. The "prob" is the probability of the binomial
        distribution.

        """

        # define a get indexes function using lambda function.
        #get_indexes = lambda x, xs: [i for (y, i) in zip(xs, range(len(xs))) if x == y]

        tt = []
        thin_sizes = [np.ones(sizes[0])]
        tt.append(sizes[0])
        for i, k in enumerate(self.sizes[1:-1]):
            bino_dist = np.random.binomial(n=1, p=prob, size=k)
            print(bino_dist)
            tt.append(np.sum(bino_dist))
            #tt = get_indexes(1, bino_dist)
            #thin_sizes[i+1] = len(tt)
            thin_sizes.append(bino_dist)
        thin_sizes.append(np.ones(sizes[-1]))
        tt.append(sizes[-1])
        mask = [np.outer(y,x)==1 for x, y in zip(thin_sizes[:-1], thin_sizes[1:])]

        return mask, tt


if  __name__ == '__main__':
    sizes = [2,4,5,2]
    net = Network(sizes)
    weights = [np.random.randn(y,x) for x, y in zip(sizes[:-1], sizes[1:])]
    mask, tt= net.dropout_size(prob=0.5)
    print(tt)
    print(weights)
    print(mask)

    for i in range(len(sizes)-1):
        print(weights[i][mask[i]].reshape(tt[i+1],tt[i]))
        B = np.zeros(np.shape(weights[i]))
        B[mask[i]] = weights[i][mask[i]]
        print(B)



