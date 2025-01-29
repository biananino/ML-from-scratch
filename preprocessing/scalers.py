import numpy as np
from abc import ABC, abstractmethod

class Scaler(ABC):
    @abstractmethod
    def fit(self,data):
        pass
    
    @abstractmethod
    def transform(self,data):
        pass
    
    @abstractmethod
    def reverse_transform(self,data):
        pass
    
    def fit_transform(self,data):
        self.fit(data)
        return self.transform(data)
    
class LinearScaler(Scaler):
    def transform(self,data):
        return self._slope*data + self._constant
    
    def reverse_transform(self,data):
        return (data - self._constant)/self._slope

class MinMaxScaler(LinearScaler):
    def __init__(self, lower = 0, upper = 1):
        self.lower = lower
        self.upper = upper
        self._slope = None
        self._constant = None
    
    def fit(self,data):
        minimum = np.min(data, axis = 0, keepdims=True)
        maximum = np.max(data, axis = 0, keepdims=True)
        self._slope = (self.upper - self.lower)/(maximum - minimum)
        self._constant = (self.lower*maximum - self.upper*minimum)/(maximum - minimum)
          
class MeanVarianceScaler(LinearScaler):
    def __init__(self, mean = 0, variance = 1):
        self.mean = mean
        self.variance = variance
        
    def fit(self,data):
        sample_mean = np.mean(data, axis = 0, keepdims = True)
        sample_var = np.var(data, axis = 0, keepdims = True)
        self._slope = np.sqrt(self.variance / sample_var)
        self._constant = ((-sample_mean) * self._slope) + self.mean
        
class MeanStdScaler(LinearScaler):
    def __init__(self, mean = 0, std = 1):
        self.mean = mean
        self.std = std
        
    def fit(self,data):
        sample_mean = np.mean(data, axis = 0, keepdims = True)
        sample_std = np.std(data, axis = 0, keepdims = True)
        self._slope = self.std / sample_std
        self._constant = ((-sample_mean) * self._slope) + self.mean
