import numpy as np

class FFNN(object):
    def __init__(self):
        self.W = None
        self.v = None
        self.sigmoid = lambda x: np.exp(x) / (np.exp(x) + 1)
        
    def train(self, X, y, l, alpha=1.0, lam=1.0, max_iter=500, e=0.001):        
        # Initialize variables
        m, n = X.shape
        self.W = np.random.rand(n+1, l)
        self.v = np.random.rand(l+1, 1)
        # Add bias term
        X = np.hstack((np.ones((m, 1)), X))
        
        # Gradient descent
        W_ = None
        v_ = None
        for i in range(max_iter):
            # Forward prop
            L = np.hstack((np.ones((m, 1)), self.sigmoid(X @ self.W)))
            y_ = self.sigmoid(L @ self.v)
            # Back prop
            d = y_ - y
            v_ = self.v - alpha * (1 / m) * ((L.T @ d) + lam * np.vstack((np.zeros(1), self.v[1:])))
            D = (d @ self.v.T) * (L * (1 - L))
            D = D[:, 1:] # Drop bias term
            W_ = self.W - alpha * (1 / m) * ((X.T @ D) + lam * np.vstack((np.zeros((1, l)), self.W[1:, :])))
            
            if np.linalg.norm(self.W - W_) < e and np.linalg.norm(self.v - v_) < e:
                self.W = W_
                self.v = v_
                break
            else:
                self.W = W_
                self.v = v_
        
    def predict(self, X):
        m, n = X.shape
        # Add bias term
        X = np.hstack((np.ones((m, 1)), X))
        # Forward prop
        L = np.hstack((np.ones((m, 1)), self.sigmoid(X @ self.W)))
        y = self.sigmoid(L @ self.v) >= 0.5
        y = np.where(y==True, 1, 0)
        return y        
