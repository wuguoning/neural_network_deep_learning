"""
Import modules
"""
import sys
import os

# Import the mnist_loader function.
module_path = os.path.abspath(os.path.join('../src/perceptrons'))
if module_path not in sys.path:
    sys.path.append(module_path)

import cost_fun
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def data_init():
    x = 1.0
    y = 0.0
    eta = 0.15
    weight = 2.0
    bias =  2.0
    iterNum = 300

    return x, y, eta, weight, bias, iterNum

def data_gen(t=0):
    x,y,eta,weight,bias,iterNum = data_init()

    obj = cost_fun.CostFunction(weight, bias)
    for i in range(iterNum):
        t += 1
        obj.GD(x, y, eta)
        neuron_output = obj.evaluate(x, y)
        yield t, neuron_output

def cross_entropy_data_gen(t=0):
    x,y,eta,weight,bias,iterNum = data_init()

    obj = cost_fun.CostFunction(weight, bias)
    for i in range(iterNum):
        t += 1
        obj.CrossEntropy(x, y, eta)
        neuron_output = obj.evaluate(x, y)
        yield t, neuron_output

def init():
    ax.set_ylim(0, 1.0)
    ax.set_xlim(0, 320)
    del xdata[:]
    del ydata[:]
    line.set_data(xdata, ydata)
    return line

def run(data):
    t, y = data
    xdata.append(t)
    ydata.append(y)
    xmin, xmax = ax.get_xlim()

    if t>= xmax:
        ax.set_xlim(xmin, 2*xmax)
        ax.figure.canvas.draw()
    line.set_data(xdata,ydata)

    return line


if __name__ == "__main__":
    fig, ax = plt.subplots(figsize=(10,8))
    line,   = ax.plot([], [], lw=2)
    ax.grid()
    xdata, ydata = [], []

    ani = animation.FuncAnimation(fig, run, data_gen, blit=False, \
                                  interval=5, repeat=False, init_func=init)
    plt.show()

    fig, ax = plt.subplots(figsize=(10,8))
    line,   = ax.plot([], [], lw=2)
    ax.grid()
    xdata, ydata = [], []

    ani = animation.FuncAnimation(fig, run, cross_entropy_data_gen, blit=False, \
                                  interval=5, repeat=False, init_func=init)
    plt.show()
