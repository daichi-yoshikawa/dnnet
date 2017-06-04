
# coding: utf-8

# In[2]:

# Authors: Daichi Yoshikawa <daichi.yoshikawa@gmail.com>
# License: BSD 3 clause

from __future__ import absolute_import

import numpy as np
from .layer import Layer


# In[3]:

class DropoutLayer(Layer):
    """Implement Dropout.

    Derived class of Layer.

    Parameters
    ----------
    drop_ratio : float
        Ratio of neurons to drop out. If it is 0.2,
        20% neurons will be dropped out and 80% neurons remain.
    mask : np.array
        Mask matrix in 2d array to dropout neurons.
    dtype : type
        Data type to use.

    Reference
    ---------
    Dropout: A Simple Way to Prevent Neural Networks from Overfitting
    https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf
    """

    def __init__(self, drop_ratio):
        self.drop_ratio = drop_ratio
        self.mask = np.array([])

    def set_dtype(self, dtype):
        self.dtype = dtype
        self.drop_ratio = dtype(self.drop_ratio)
        self.mask = self.mask.astype(dtype)

    def get_type(self):
        return 'dropout'

    def set_parent(self, parent):
        Layer.set_parent(self, parent)
        self.shape = parent.shape
        self.mask = np.ones(self.shape, dtype=self.dtype)

    def forward(self, x):
        self.mask = np.random.rand(self.mask.shape[0]) <= (1. - self.drop_ratio)
        self.fire = self.mask * x
        self.child.forward(self.fire)

    def backward(self, dy):
        self.backfire = self.mask * dy
        self.parent.backward(self.backfire)

    def predict_to_eval(self, x):
        self.fire = (1. - self.drop_ratio) * x
        return self.child.predict_to_eval(self.fire)

    def predict(self, x):
        self.fire = (1. - self.drop_ratio) * x
        return self.child.predict(self.fire)

    def finalize_training(self, x):
        self.fire = (1. - self.drop_ratio) * x
        self.child.finalize_training(self.fire)


# In[ ]:



