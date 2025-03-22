import numpy as np

class KNN_Regressor:
    def __init__(self, k=3, weighted=False):
        self.k = k
        self.weighted = weighted

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def distance(self, X1, X2):
        return np.sqrt(np.sum((X1 - X2) ** 2))
    
    def single_predict(self, x):
        
        distances = [self.distance(x, x_train) for x_train in self.X_train]
        kth_points = np.argsort(distances)[:self.k]
        if self.weighted:
            
            k_distances = np.array([distances[i] for i in kth_points])
            k_values = np.array([self.y_train[i] for i in kth_points])
            weights = 1 / (k_distances ** 2 + 1e-8)  

            return np.sum(weights * k_values) / np.sum(weights)
        else:
            k_values = [self.y_train[i] for i in kth_points]
            return np.mean(k_values)
        
    def predict(self, X):
        return np.array([self.single_predict(x) for x in X])
    
    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean((y_pred - y) ** 2)
    
    
        
        