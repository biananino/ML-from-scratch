import numpy as np
from abc import ABC, abstractmethod

class Solver(ABC):
    @abstractmethod
    def train(self,model,training_data,validation_data = None):
        pass

class MiniBatchSGD(Solver):
    def __init__(self,batch_size = 256,learning_rate = 1e-5):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        
    def train(self,model,training_data,validation_data = None, iterations = 100, seed = None):
        rng = np.random.default_rng(seed)
        X_train = training_data[0]
        y_train = training_data[1]
        num_inputs = X_train.shape[0]
        for _ in range(iterations):
            mask = rng.choice(num_inputs, size = self.batch_size)
            X_batch = X_train[mask,:]
            y_batch = y_train[mask]
            gradients = model._backward_pass(X_batch,y_batch)            
            for layer, gradients in zip(model._layers[1:],gradients):
                weight_grad = gradients[1]
                bias_grad = gradients[2]
                layer.weights += -self.learning_rate*weight_grad
                layer.bias += -self.learning_rate*bias_grad
     
class MiniBatchPSGD(Solver):
    def __init__(self,batch_size = 256,learning_rate = 1e-5):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        
    def _batch_number(self,num_inputs):
        if num_inputs % self.batch_size == 0:
            batches = num_inputs // self.batch_size
        else:
            batches = 1 + (num_inputs // self.batch_size)
        return batches
        
    def train(self,model,training_data,validation_data = None, epochs = 100, seed = None):
        rng = np.random.default_rng(seed)
        X_train = training_data[0]
        y_train = training_data[1]
        num_inputs = X_train.shape[0]
        batches = self._batch_number(num_inputs)
        labels = np.arange(num_inputs)
        for _ in range(epochs):
            rng.shuffle(labels)
            for batch in range(batches):
                start_mask = self.batch_size * batch
                end_mask = self.batch_size * (batch + 1)
                mask = labels[start_mask:end_mask]
                X_batch = X_train[mask,:]
                y_batch = y_train[mask]
                gradients = model._backward_pass(X_batch,y_batch)            
                for layer, gradients in zip(model._layers[1:],gradients):
                    weight_grad = gradients[1]
                    bias_grad = gradients[2]
                    layer.weights += -self.learning_rate*weight_grad
                    layer.bias += -self.learning_rate*bias_grad
                
class GradientDescent(Solver):
    def __init__(self,learning_rate = 1e-5):
        self.learning_rate = learning_rate
        
    def train(self,model,training_data,validation_data = None, epochs = 10):
        X_train = training_data[0]
        y_train = training_data[1]
        for epoch in range(epochs):
            gradients = model._backward_pass(X_train,y_train)            
            for layer, gradients in zip(model._layers[1:],gradients):
                weight_grad = gradients[1]
                bias_grad = gradients[2]
                layer.weights += -self.learning_rate*weight_grad
                layer.bias += -self.learning_rate*bias_grad 
