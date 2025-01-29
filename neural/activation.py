import numpy as np
from abc import ABC, abstractmethod

class Activation(ABC):
    @abstractmethod
    def __call__(self,X):
        pass
    
    @abstractmethod
    def gradient(self,upGrad,X):    
        pass
    
class Linear(Activation):
    def __call__(self,X):
        return X
    
    def gradient(self,upGrad,X):
        return upGrad
    
    
class Relu(Activation):
    def __call__(self,X):
        return np.maximum(0,X)
    
    def gradient(self,upGrad,X):
        return np.where(X > 0,upGrad,0)
    
class LeakyRelu(Activation):
    def __init__(self,alpha = 0.001):
        self.alpha = alpha
        
    def __call__(self,X):
        return np.where(X > 0, X, X * self.alpha)
    
    def gradient(self,upGrad,X):
        return np.where(X > 0, upGrad, upGrad * self.alpha)

class Sigmoid(Activation):
    @staticmethod
    def _positive_sigmoid(x):
        denominator = (1 + np.exp(-x))
        return 1/denominator
    
    @staticmethod
    def _negative_sigmoid(x):
        exponential = np.exp(x)
        return exponential/(1 + exponential)
    
    def __call__(self,X):
        positives = X >= 0
        negatives = ~positives
        result = np.empty_like(X, dtype=np.float64)
        result[positives] = self._positive_sigmoid(X[positives])
        result[negatives] = self._negative_sigmoid(X[negatives])
        return result
    
    def gradient(self,upGrad,X):
        output = self.__call__(X)
        return upGrad*output*(1-output) 