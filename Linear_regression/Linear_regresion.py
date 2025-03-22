import numpy as np

class MutivariantLinearRegression:
    def __init__(self,learningrate = 0.01,epocs = 1000):
        self.learningrate = learningrate
        self.epocs = epocs
        self.weights = None
        self.bias = None

    def normalize(self,x):
        mean = np.mean(x,axis = 0)
        std = np.std(x,axis = 0)
        std[std == 0] = 1
        return (x-mean)/std
    
    def fit(self,x,y):
        x = np.array(x)
        y = np.array(y)

        '''
            x is the size of [examples,features]
            y is the size of [examples,]
        '''
        features_size = x.shape[1]
        examples_size = x.shape[0]

        x = self.normalize(x)

        weights = np.zeros(features_size) #this gives the size of [features,]
        bias = 0
        loss = []
        for i in range(self.epocs):
            y_predicted = np.dot(x,weights) + bias #x is size of [examples,features] and weights is size of [features,] then out put is [examples,] which is the size of y
            dw = (1/examples_size)*np.dot(x.T,(y_predicted-y)) #x.T is the size of [features,examples] and y_predicted-y is the size of [examples,] then the output is [features,] we can use this to update the weights
            db = (1/examples_size)*np.sum(y_predicted-y) #this is the size of [1,] which is the size of bias

            weights = weights - self.learningrate*dw    #updating the weights
            bias = bias - self.learningrate*db          #updating the bias
            loss.append(np.mean(np.square(y_predicted-y)))

        self.weights = weights
        self.bias = bias
        return loss

    def predict(self,x):
        x = np.array(x)
        x = self.normalize(x)
        return np.dot(x,self.weights) + self.bias #x is the size of [examples,features] and weights is the size of [features,] then out put is [examples,] which is the size of y
