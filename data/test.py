import pickle
import gzip
import numpy
import sys

with gzip.open('mnist.pkl.gz', 'rb') as f:
    if sys.version_info.major > 2:
        train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
    else:
        train_set, valid_set, test_set = pickle.load(f)

train_data = zip(train_set[0], train_set[1])
for x, y in train_data:
    print(x,y)
print(len(valid_set[0]))
print(len(test_set[0]))
