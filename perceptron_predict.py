# CSE 3521 Programming Assignment 8
# author: Ron Chen.8336

import sys
import numpy as np

def predict(argv:list):
    weight = []
    with open(argv[1]) as fid:
        for line in fid:
            weight.append(float(line.rstrip('\n')))

    W = np.array(weight)
    W = W.reshape((W.size, 1))

    data = []
    with open(argv[2]) as fid:
        for line in fid:
            line = line.rstrip('\n').split(',')
            line = [int(i) for i in line]
            data.append(line)
    
    d = data.pop(0).pop(0)
    X = np.array(data)
    Y = X[:, d]
    X = np.delete(X, d, 1)
    X = np.hstack((X, np.ones((Y.size, 1)))).T
    
    P = (W.T @ X).reshape(Y.size)
    np.place(P, P>=0, [1])
    np.place(P, P<0, [0])
    print(P)
    print(Y)

predict(sys.argv)
