import numpy as np
from abc import ABC, abstractmethod

from .activation import *
from .regularizer import *
from .initializer import *

class layer(ABC):
    @abstractmethod
    def __init__(self,size,activation,regularizer):
        pass
    
    def __repr__(self):
        return f"{self.__class__.__name__}: {self.size}"
    
    @abstractmethod
    def _forward_pass(self,X):
        pass

class inputLayer(layer):
    def __init__(self,size):
        self.size = size
        
    def _forward_pass(self,X):
        return X    
        
class denseLayer(layer):
    def __init__(self, size, activation = linear(), weight_regularizer = nullRegularizer(), 
                 bias_regularizer = nullRegularizer(), weight_initializer = normalInitializer(),
                 bias_initializer = zeroInitializer()):        
        self.size = size
        self.activation = activation
        self.bias_regularizer = bias_regularizer
        self.weight_regularizer = weight_regularizer
        self.weights = None
        self.bias = None
        self.weight_initializer = weight_initializer
        self.bias_initializer = bias_initializer
        self._input_dims = None
        self._output_dims = size
        
    def __repr__(self):
        return  f"Dense layer: {self.size}"
        
    def _forward_pass(self, X):
        scores = np.dot(X,self.weights) + self.bias
        return self.activation(scores)

    def _gradient(self,upGrad,X,score):
        upGrad = self.activation.gradient(upGrad,score)
        n_inputs = upGrad.shape[0]
        weights_grad = np.dot(X.T,upGrad) + self.weight_regularizer.gradient(self.weights)/n_inputs
        bias_grad = np.dot(upGrad.T,np.ones(n_inputs))  + self.bias_regularizer.gradient(self.bias)/n_inputs
        downGrad = np.dot(upGrad,self.weights.T)
        return downGrad, weights_grad, bias_grad
    
    def regularization_loss(self):
        return self.weight_regularizer(self.weights) + self.bias_regularizer(self.bias)
    
    def predict(self,X):
        return self._forward_pass(X)
    
    def _initialize(self,seed = None):
        rng = np.random.default_rng(seed)
        bias_shape = self.size
        weights_shape = (self._input_dims,self.size)
        self.bias = self.bias_initializer(bias_shape,rng)
        self.weights = self.weight_initializer(weights_shape,rng)
