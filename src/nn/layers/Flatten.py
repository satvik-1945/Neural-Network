import numpy as np
 
class Flatten():
    def __init__(self, input_shape=None):
        self.input_shape = input_shape
        self.original_shape = None
 
    def forward(self, x):
        self.original_shape = x.shape
        return x.reshape(x.shape[0], -1)
 
    def backward(self, x, grad, learning_rate):
        return grad.reshape(self.original_shape)
 
    @property
    def output_size(self):
        if self.input_shape:
            return np.prod(self.input_shape)
        return None
 
#sequential
import pickle
 
import numpy as np
from keras.src.metrics.accuracy_metrics import accuracy