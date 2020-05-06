# CSE 3521 Programming Assignment 8
# author: Ron Chen.8336

import sys
import numpy as np

def train(X:np.ndarray, Y:np.ndarray) -> np.ndarray:    
    n, m = X.shape
    np.place(Y, Y==0, [-1])
    W = np.zeros((n, 1))
    alpha = 1

    for j in range(100):
        converge = True
        for i in range(m):
            y = 1 if (W.T @ X[:, i]) >= 0 else -1
            if y != Y[i]:
                W = W + alpha * (X[:, i].reshape((n, 1)) * Y[i])
                converge = False

        if converge:
            break

    return W

def main(argv:list):
    data = []
    with open(argv[1]) as fid:
        for line in fid:
            line = line.rstrip('\n').split(',')
            line = [int(i) for i in line]
            data.append(line)

    d = data.pop(0).pop(0)
    X = np.array(data)
    Y = X[:, d]
    X = np.delete(X, d, 1)
    X = np.hstack((X, np.ones((Y.size, 1)))).T

    W = train(X, Y)
    print(W)

    weight = open('weight.txt', 'w+')
    for w in np.nditer(W):
        weight.write(str(w) + '\n')
    weight.close()

main(sys.argv)
