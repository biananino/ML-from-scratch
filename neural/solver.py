import numpy as np
from abc import ABC, abstractmethod

class solver(ABC):
    @abstractmethod
    def train(self,model,training_data,validation_data = None):
        pass

class miniBatchSGD(solver):
    def __init__(self,batch_size = 256,learning_rate = 1e-5):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        
    def train(self,model,training_data,validation_data = None, iterations = 100, seed = None):
        rng = np.random.default_rng(seed)
        X_train = training_data[0]
        y_train = training_data[1]
        num_inputs = X_train.shape[0]
        for _ in range(iterations):
            labels = rng.choice(num_inputs, size = self.batch_size)
            X_batch = X_train[labels,:]
            y_batch = y_train[labels]
            gradients = model._backward_pass(X_batch,y_batch)            
            for layer, gradients in zip(model._layers[1:],gradients):
                weight_grad = gradients[1]
                bias_grad = gradients[2]
                layer.weights += -self.learning_rate*weight_grad
                layer.bias += -self.learning_rate*bias_grad   
                

class gradientDescent(solver):
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
