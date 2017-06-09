
# coding: utf-8

# In[1]:

# Authors: Daichi Yoshikawa <daichi.yoshikawa@gmail.com>
# License: BSD 3 clause

from __future__ import absolute_import

import numpy as np

from .layer import Layer
from ..utils.conv_utils import im2col


# In[ ]:

class PoolingLayer(Layer):
    def __init__(self, pool_shape):
        self.pool_shape = pool_shape
        
    def get_type(self):
        return 'pooling'
    
    def forward(self, x):
        pass
    
    def backward(self, dy):
        pass
    
    def predict_to_eval(self, x):
        pass
    
    def predict(self, x):
        pass
    
    def finalize_training(self, x):
        pass

