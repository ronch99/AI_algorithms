#Programming Assignment 7: Gaussian Mixture Models
#author Ron Chen.8336

import sys
import numpy as np

from scipy.stats import multivariate_normal

def GMM(k: int, X: np.ndarray):
    size, dimension = X.shape
    normalize = lambda x: x / np.sum(x)
    calculate_cov = (lambda p, x, u:
                     p * np.matmul((x-u).reshape(dimension, 1),
                                   (x-u).reshape(1, dimension)))
    AIC = (lambda n, l:
           2 * (n - 1 + n * dimension + n * (dimension ** 2)) - 2 * l)
    
    # Initialize parameters
    Pi = np.array([(1 / k) for i in range(k)])
    Mu = X[0:k]
    Sigma = np.ones((k, dimension, dimension))
    P = np.ndarray((k, size))
    L = 0.0

    Pi_t = np.ndarray(k)
    Mu_t = np.ndarray((k, dimension))
    Sigma_t = np.ndarray((k, dimension, dimension))
    
    for _ in range(1000):
        # Evaluate probabilities
        for j in range(k):
            P[j] = multivariate_normal.pdf(X, mean=Mu[j],
                                           cov=Sigma[j], allow_singular=True)
            P[j] = P[j] * Pi[j]
        
        # Calculate log likelihood
        L = np.sum(np.log(np.apply_along_axis(np.sum, 0, P)))
        # Normalize probabilities
        P = np.apply_along_axis(normalize, 0, P)
        
        # Maximize likelihood and update arguments
        for i in range(k):
            Pi_t[i] = np.sum(P[i])
            Mu_t[i] = np.apply_along_axis(np.matmul, 0, X, P[i]) / Pi_t[i]
            
            cov = np.zeros((dimension, dimension))
            for j in range(size):
                cov = cov + calculate_cov(P[i, j], X[j], Mu_t[i])
            Sigma_t[i] = cov / Pi_t[i]
        Pi_t = Pi_t / size

        # Check for convergence
        converge = (np.max(np.abs(Pi - Pi_t)) < 0.000001 and
                    np.max(np.abs(Mu - Mu_t)) < 0.000001 and
                    np.max(np.abs(Sigma - Sigma_t)) < 0.000001)
        
        Pi = np.copy(Pi_t)
        Mu = np.copy(Mu_t)
        Sigma = np.copy(Sigma_t)
        if converge:
            break
    
    return (AIC(k, L), Pi, Mu, Sigma, P)

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

    c = 0
    a_t = sys.maxsize
    best = (0, 0, 0, 0)
    for k in range(int(argv[2]), int(argv[3])+1):
        a, pi, mu, sigma, p = GMM(k, X)
        print(k, a)
        if a < a_t:
            a_t = a
            c = k
            best = (pi, mu, sigma, p)
    
    print('Best cluster #:', c)
    fid = open('GMM.txt', 'w+')
    fid.write('PI:\n'+np.array_str(best[0])+'\n\n')
    fid.write('MU:\n'+np.array_str(best[1])+'\n\n')
    fid.write('SIGMA:\n'+np.array_str(best[2])+'\n\n')
    fid.write('Cluster:\n'+np.array_str(np.apply_along_axis(np.argmax, 0, best[3]))+'\n')
    fid.close()

main(sys.argv)
