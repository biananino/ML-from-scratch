import numpy as np
from abc import ABC, abstractmethod

from .layer import *
from .initializer import *
from .regularizer import *
from .loss import *

class Model(ABC):
    @abstractmethod
    def __init__(self):
        pass
    
    @abstractmethod
    def _forward_pass(self,X):
        pass
    
    @abstractmethod
    def _backward_pass(self,layer,upGrad,X,y):
        pass
    
    @property
    @abstractmethod
    def layers(self):
        pass
    
    
class SequentialModel(Model):
    def __init__(self):
        self._layers = []
        self._loss = None
        
    def __repr__(self):
        return self._layers + [loss]
    
    def predict(self,X):
        return self._forward_pass(X)[-1]
    
    def add(self,layer):
        self._layers.append(layer)
    
    @property
    def loss(self):
        return self._loss
    
    @loss.setter
    def loss(self,loss):
        self._loss = loss
            
    @property
    def layers(self):
        return self._layers
    
    def _initialize(self, seed = None):
        for previous_layer, layer in zip(self._layers,self._layers[1:]):
            layer._input_dims = previous_layer.size
            layer._initialize(seed)   
    
    def _forward_pass(self,X):
        scores = []
        for layer in self._layers:
            X = layer._forward_pass(X)
            scores.append(X)
        return scores
    
    def _backward_pass(self,X,y):
        gradients = []
        scores = self._forward_pass(X)
        y_pred = scores[-1]
        upGrad = self._loss.gradient(y,y_pred)
        gradients.append(upGrad)
        for layer, previous_score, score in zip(self._layers[:0:-1],scores[-2::-1], scores[::-1]):
            gradient = layer._gradient(upGrad,previous_score, score)
            gradients.append(gradient)
            upGrad = gradient[0]
        return gradients[::-1]
