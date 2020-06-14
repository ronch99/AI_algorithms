import numpy as np

class logisticRegression(object):        
    def __init__(self):
        self.w = None
        self.sigmoid = lambda x: np.exp(x) / (np.exp(x) + 1)
        
    def train(self, X, y, alpha=1.0, lam=1.0, max_iter=500, e=0.001):
        # Define objective function and its gradient
        J = (lambda w, X, y, lam: 
             - (1 / X.shape[0]) * ((y.T) @ np.log(self.sigmoid(X @ w)) + ((1 - y).T) @ np.log(1 - self.sigmoid(X @ w))) 
             + (lam / (2 * X.shape[0])) * (w[1:].T @ w[1:]) )
        grad = (lambda w, X, y, lam: 
                (1 / X.shape[0]) * (X.T @ (self.sigmoid(X @ w) - y)) + (lam / X.shape[0]) * np.vstack((np.zeros(1), w[1:])) )
        
        # Initialize variables
        m, n = X.shape
        self.w = np.zeros((n + 1, 1))
        # Add bias term
        X = np.hstack((np.ones((m, 1)), X))
        
        # Gradient descent
        w_ = None
        for i in range(max_iter):
            w_ = self.w - alpha * grad(self.w, X, y, lam)
            if np.linalg.norm(self.w - w_) < e:
                self.w = w_
                break
            else:
                self.w = w_
        # Compute objective function
        return J(self.w, X, y, lam)
    
    def predict(self, X):
        m, n = X.shape
        # Add bias term
        X = np.hstack((np.ones((m, 1)), X))
        y = self.sigmoid(X @ self.w) >= 0.5
        y = np.where(y==True, 1, 0)
        return y
