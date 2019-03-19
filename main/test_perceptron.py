import os
import sys
import numpy as np

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from src.perceptrons import BoolPerceptron

x = [[0,0],[0,1],[1,0],[1,1]]
print(x)
percep = BoolPerceptron()
for items in x:
    print(percep.booland(items))
for items in x:
    print(percep.boolnand(items))
for items in x:
    print(percep.boolor(items))
