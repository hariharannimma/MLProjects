import numpy as np

class LinearRegression():
    
    def __init__(self, lr = 0.15, n_iter=1000):
        self.lr =  lr
        self.n_iter = n_iter
        self.weights = None
        self.bais = None
           
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bais = 0
        
        for _ in range(self.n_iter):
            y_pred = np.dot(X, self.weights) + self.bais
            
            dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
            db = (1/n_samples) * np.sum(y_pred-y)
            
            self.weights -= self.lr * dw
            self.bais -= self.lr * db
            
        w = self.weights
        b = self.bais
        print(f"w value is: {w}, b value is: {b}")
        
    def predict(self, X):
        y_pred = np.dot(X, self.weights) + self.bais
        return y_pred