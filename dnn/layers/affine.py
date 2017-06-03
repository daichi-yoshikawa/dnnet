
# coding: utf-8

# In[17]:

# Authors: Daichi Yoshikawa <daichi.yoshikawa@gmail.com>
# License: BSD 3 clause

from __future__ import absolute_import

import numpy as np

from .layer import Layer
from ..training.random_weight import RandomWeight, RandomWeightFactory
from ..training.random_weight import DefaultRandomWeight, Xavier, He


# In[12]:

class AffineLayer(Layer):
    """Implement affine transform of matrix.

    Derived class of Layer.

    Parameters
    ----------
    parent_shape : tuple
        Shape of parent layer's neurons.
    shape : tuple
        Shape of this layer's neurons.
    w : np.array
        Weight in 2d array.
    dw : np.array
        Gradient of weight in 2d array.
    x : np.array
        Extended parent layer's output in 2d array.
        This consists of original parent layer's output and bias term.
    """
    def __init__(self, shape, random_weight=RandomWeight.Type.default):
        self.parent_shape, self.shape = shape
        self.w = DefaultRandomWeight().get(self.parent_shape, self.shape)
        self.w = np.r_[np.zeros((1, self.shape)), self.w]
        self.dw = np.zeros_like(self.w, dtype=self.w.dtype)
        self.x = None

    def set_dtype(self, dtype):
        self.dtype = dtype
        self.w = self.w.astype(dtype)
        self.dw = self.dw.astype(dtype)

    def get_type(self):
        return 'affine'

    def has_weight(self):
        return True

    def forward(self, x):
        self.__forward(x)
        self.child.forward(self.fire)

    def backward(self, dy):
        batch_size = self.x.shape[0]
        self.dw = self.dtype(1.) / batch_size * np.dot(self.x.T, dy)        
        self.backfire = np.dot(dy, self.w[1:, :].T)
        self.parent.backward(self.backfire)

    def predict_to_eval(self, x):
        self.__forward(x)
        return self.child.predict_to_eval(self.fire)

    def predict(self, x):
        self.__forward(x)
        return self.child.predict(self.fire)

    def finalize_training(self, x):
        self.__forward(x)
        self.child.finalize_training(self.fire)

    def __forward(self, x):
        # Add bias terms.
        self.x = np.c_[np.ones((x.shape[0], 1), dtype=self.dtype), x]
        self.fire = np.dot(self.x, self.w)


# In[ ]:



