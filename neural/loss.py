import numpy as np
from abc import ABC, abstractmethod

class loss(ABC):  
    
    @abstractmethod
    def __call__(self,y_true,y_pred):
        pass
    
    @abstractmethod
    def gradient(self,y_true,y_pred):
        pass
    
class mse(loss):
    def __call__(self,y_true,y_pred):
        RSS = (1/2)*(y_true - y_pred)**2
        mse = np.mean(RSS)
        return  mse
    
    def gradient(self,y_true,y_pred):
        num_inputs = y_true.shape[0]
        gradient = -(y_true - y_pred)/num_inputs
        return gradient

class SparseCrossEntropy_withlogits(loss):
    def __call__(self,y_true,y_pred):
        num_inputs = y_true.shape[0]
        max_score = np.max(y_pred, axis = 1, keepdims = True)
        norm_constant = np.sum(np.exp(y_pred - max_score), axis = 1, keepdims= True)
        log_probs = y_pred - max_score - np.log(norm_constant)
        correct_probabilities = log_probs[range(num_inputs),y_true]
        loss = -np.mean(correct_probabilities)
        return loss
    
    def gradient(self,y_true,y_pred):
        num_inputs = y_true.shape[0]
        exp_scores = np.exp(y_pred - np.max(y_pred, axis = 1, keepdims = True)) #Numerical stability
        exp_sums = np.sum(exp_scores, axis = 1, keepdims = True)
        probabilities = exp_scores/exp_sums
        gradient = probabilities
        gradient[range(num_inputs),y_true] += - 1
        gradient /= num_inputs
        return gradient
