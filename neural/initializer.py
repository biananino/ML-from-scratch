import numpy as np
from abc import ABC, abstractmethod

class initializer(ABC):
    @abstractmethod
    def __call__(self,shape,seed = None):
        pass
    
class zeroInitializer(initializer):
        def __call__(self,shape,seed = None):
            return np.zeros(shape)

class normalInitializer(initializer):
    def __init__(self,scale = 1e-3):
        self.scale = scale
        
    def __call__(self,shape,seed = None):
        rng = np.random.default_rng(seed)
        return rng.standard_normal(shape)*self.scale
