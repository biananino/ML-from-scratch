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
    
    def gradient(self, upGrad, X):
        return upGrad
    
    
class Relu(Activation):
    def __call__(self,X):
        return np.maximum(0,X)
    
    def gradient(self,upGrad,X):
        return np.where(X > 0, upGrad, 0)
    
class LeakyRelu(Activation):
    def __init__(self,alpha = 0.001):
        self.alpha = alpha
        
    def __call__(self,X):
        return np.where(X > 0, X, X * self.alpha)
    
    def gradient(self,upGrad,X):
        return np.where(X > 0, upGrad, upGrad * self.alpha)
