
# coding: utf-8

# In[ ]:


# Authors: Daichi Yoshikawa <daichi.yoshikawa@gmail.com>
# License: BSD 3 clause

from __future__ import absolute_import

import numpy as np

from .layer import Layer
from ..utils.conv_utils import im2col

class PoolingLayer(Layer):
    def __init__(self, pool_shape):
        self.pool_shape = pool_shape

    def get_type(self):
        return 'pooling'

    def set_parent(self, parent):
        Layer.set_parent(self, parent)

    def forward(self, x):
        pass

    def backward(self, dy):
        pass

    def predict(self, x):
        pass

    def finalize_training(self, x):
        pass

    def __forward(self, x):
        pass

    def __backward(self, dy):
        pass

