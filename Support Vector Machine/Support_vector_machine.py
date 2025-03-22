import numpy as np

class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def init_params(self, X):
        n_features = X.shape[1] 
        self.w = np.random.randn(n_features) * 0.01
        self.b = 0

    def calculate_cost(self, X, y):
        n_samples = X.shape[0]
        distances = y * (np.dot(X, self.w) + self.b)
        hinge_loss = np.maximum(0, 1 - distances)
        cost = np.mean(hinge_loss)  
        regularized_cost = 0.5 * self.lambda_param * np.dot(self.w, self.w)
        total_cost = regularized_cost + cost
        return total_cost
    
    def compute_gradients(self, X, y):
        n_samples = X.shape[0]
        distances = y * (np.dot(X, self.w) + self.b)
        mask = distances < 1
        
        dw = self.lambda_param * self.w  
        X_masked = X[mask]
        y_masked = y[mask]
        dw -= np.dot(X_masked.T, y_masked) / n_samples
        db = -np.sum(y_masked) / n_samples
        
        return dw, db
    
    def fit(self, X, y):
        if self.w is None:
            self.init_params(X)  
        
        costs = []
        
        for i in range(self.n_iters):  
            dw, db = self.compute_gradients(X, y) 
            
            self.w -= self.lr * dw
            self.b -= self.lr * db
            
            cost = self.calculate_cost(X, y) 
            costs.append(cost)
        
        return costs
    
    def predict(self, X):
        approx = np.dot(X, self.w) + self.b
        return np.sign(approx)