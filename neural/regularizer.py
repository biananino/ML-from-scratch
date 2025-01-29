import numpy as np
from abc import ABC, abstractmethod

class Regularizer(ABC):
    @abstractmethod
    def __init__(self):
        pass
    
    @abstractmethod
    def __call__(self,W):
        pass
    
    @abstractmethod
    def gradient(self,W):
        pass
    
class NullRegularizer(Regularizer):
    def __init__(self):
        pass
    
    def __call__(self,W):
        return 0
    
    def gradient(self,W):
        return 0

class L1(Regularizer):
    def __init__(self,alpha):
        self.alpha = alpha
        
    def __call__(self,W):
        return self.alpha * np.sum(np.abs(W))
    
    def gradient(self,W):
        return np.sign(W)*self.alpha
    
class L2(Regularizer):
    def __init__(self,alpha):
        self.alpha = alpha
        
    def __call__(self,W):
        return self.alpha * (1/2) * np.sum(W**2)
    
    def gradient(self,W):
        return W*self.alpha
    
class L1L2(Regularizer):
    def __init__(self,alpha_l1,alpha_l2):
        self.l1 = L1(alpha_l1)
        self.l2 = L2(alpha_l2)
        
    def __call__(self,W):
        return self.l1(W) + self.l2(W)
    
    def gradient(self,W):
        return self.l1.gradient(W) + self.l2.gradient(W)
