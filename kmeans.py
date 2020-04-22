#Programming Assignment 7: K-means Clustering
#author Ron Chen.8336

import sys
import random
import numpy as np
import matplotlib.pyplot as plt

def Kmeans(k: int, X: np.ndarray) -> float:
    size, dimension = X.shape
    dist = lambda x, y: np.sum((x - y) ** 2)
    argmin_dist = lambda x, M: np.argmin(np.apply_along_axis(dist, 1, M, x))
    
    # Randomly partition into k groups
    P = np.array([random.randint(0, k-1) for i in range(size)])
    
    # Compute centers for each group
    M = np.zeros((k, dimension))
    S = 0.0
    for i in range(k):
        group = X[np.where(P == i)]
        if group.size > 0:
            m = np.apply_along_axis(np.mean, 0, group)
            M[i] = m
            S = S + np.sum(np.apply_along_axis(dist, 1, group, m))
    
    # Update centers until no change in cluster assignment
    for i in range(100):
        P2 = np.apply_along_axis(argmin_dist, 1, X, M)
        
        S = 0.0
        for j in range(k):
            group = X[np.where(P2 == j)]
            if group.size > 0:
                m = np.apply_along_axis(np.mean, 0, group)
                M[j] = m
                S = S + np.sum(np.apply_along_axis(dist, 1, group, m))

        if np.array_equal(P, P2):
            break
        else:
            P = P2

    print('k =', k, ',', S/k)
    
    # Output file
    fname = 'Kmeans' + str(k) + '.txt'
    fid = open(fname, 'w+')
    fid.write('Centroids:\n'+np.array_str(M)+'\n\n')
    fid.write('Clusters:\n'+np.array_str(P)+'\n\n')
    fid.close()
        
    return S / k
    

def main(argv: list):
    if int(argv[2]) <= 0 or int(argv[3]) > 100:
        print('Invalid range')
        sys.exit()
    
    data = []
    with open(argv[1]) as fid:
        for line in fid:
            line = line.split(',')
            line = [float(x) for x in line]
            data.append(line)

    X = np.array(data).T
    np.set_printoptions(threshold=sys.maxsize)

    c = []
    for k in range(int(argv[2]), int(argv[3])+1):
        c.append(Kmeans(k, X))

    K = np.arange(int(argv[2]), int(argv[3])+1, 1)
    C = np.array(c)
    fig = plt.figure()
    plt.plot(K, C, marker='o')
    ax = fig.gca()
    ax.set_xticks(K)
    ax.set_xticklabels(K)
    plt.savefig('kmeans.png')
        

main(sys.argv)
