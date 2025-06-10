import numpy as np
from tensorflow.python.keras.backend import learning_phase
 
 
class Dense():
    def __init__(self, units, activation='relu', input_shape=None):
        self.units = units
        self.activation = activation
        self.input_shape = input_shape
        self.weights = None
        self.biases = None
        self.last_input = None
 
    def initialize_weights(self, input_size):
        if self.activation == 'relu':
            self.weights = np.random.randn(input_size, self.units) * np.sqrt(2.0 / input_size)
        else:
            self.weights = np.random.randn(input_size, self.units) * np.sqrt(1.0 / input_size)
 
        self.biases = np.zeros((1, self.units))
 
    def forward(self, x):
        self.last_input = x
        z = np.dot(x, self.weights) + self.biases
 
        if self.activation == 'relu':
            return np.maximum(0, z)
        elif self.activation == 'softmax':
            exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
            return exp_z / np.sum(exp_z, axis=1, keepdims=True)
        else:
            return z
 
    def backward(self, x, grad, learning_rate):
        m = x.shape[0]
        z = np.dot(x, self.weights) + self.biases
 
        if self.activation == 'relu':
            activation_grad = (z > 0).astype(float)
            grad = grad * activation_grad
        elif self.activation == 'softmax':
            pass
 
        dW = np.dot(x.T,grad)/m
        db = np.sum(grad, axis=0,keepdims=True)/m
        dx = np.dot(grad, self.weights.T)
 
        self.weights -= learning_rate * dW
        self.biases -= learning_rate *db
 
        return dx