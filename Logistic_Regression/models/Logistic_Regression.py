import numpy as np

class LogisticRegression:
    def __init__(self,learning_rate=0.01,iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = None

    def normalize(self,X):
        mean = np.mean(X,axis=0)
        std = np.std(X,axis=0)
        std[std==0] = 1
        X = (X-mean)/std
        return X
    
    def sigmoid(self,z):
        return 1/(1+np.exp(-z))
    
    def loss(self,h,y):
        return (-y*np.log(h)-(1-y)*np.log(1-h)).mean()
    
    def fit(self,X,y):
        '''
        X input size is [Samples,Features]
        y output size is [Samples,]
        '''	

        X = self.normalize(X)

        self.weights = np.zeros(X.shape[1])
        self.bias = 0   

        loss = []

        for i in range(self.iterations):
            z = np.dot(X,self.weights)+self.bias # [Samples,Features] * [Features,] = [Samples,]

            h = self.sigmoid(z) # [Samples,]

            dw = np.dot(X.T,h-y)/y.size # [Features,Samples] * [Samples,] = [Features,] we get Transpose of X because we want to multiply each feature with the error
            #so that we get the gradient of each feature

            db = np.sum(h-y)/y.size

            self.weights -= self.learning_rate*dw
            self.bias -= self.learning_rate*db

            loss.append(self.loss(h,y))

        return loss
    
    def predict(self,X):
        X = self.normalize(X)
        z = np.dot(X,self.weights)+self.bias
        return self.sigmoid(z) > 0.5
    
    def accuracy(self,X,y):
        return np.mean(self.predict(X)==y)


