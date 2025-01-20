import numpy as np
from abc import ABC, abstractmethod

class activation(ABC):
    @abstractmethod
    def __call__(self,X):
        pass
    
    @abstractmethod
    def gradient(self,upGrad,X):    
        pass
    
class linear(activation):
    def __call__(self,X):
        return X
    
    def gradient(self, upGrad, X):
        return upGrad
    
    
class relu(activation):
    def __call__(self,X):
        return np.maximum(0,X)
    
    def gradient(self,upGrad,X):
        return np.where(X > 0, upGrad, 0)
    
class leakyRelu(activation):
    def __init__(self,alpha = 0.001):
        self.alpha = alpha
        
    def __call__(self,X):
        return np.where(X > 0, X, X * self.alpha)
    
    def gradient(self,upGrad,X):
        return np.where(X > 0, upGrad, upGrad * self.alpha)
