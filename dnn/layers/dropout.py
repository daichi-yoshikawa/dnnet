
# coding: utf-8

# In[1]:

# Authors: Daichi Yoshikawa <daichi.yoshikawa@gmail.com>
# License: BSD 3 clause

from __future__ import absolute_import

import numpy as np
from .layer import Layer
from ..utils import is_multi_channels_image
from ..utils import flatten, unflatten


# In[2]:

class DropoutLayer(Layer):
    """Implement Dropout.

    Derived class of Layer.
    Mask to drop out some neurons is made by shuffling 1d array.
    Number of neurons dropped out is always constant.

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

    def get_type(self):
        return 'dropout'

    def set_parent(self, parent):
        Layer.set_parent(self, parent)
        self.output_shape = self.input_shape

        input_size = np.prod(self.input_shape)
        self.mask = np.arange(input_size).reshape(self.input_shape)

    def forward(self, x):
        np.random.shuffle(self.mask.reshape(self.mask.size))
        self.fire = (self.mask >= int(self.drop_ratio*self.mask.size)) * x
        self.child.forward(self.fire)

    def backward(self, dy):
        self.backfire = (self.mask >= int(self.drop_ratio*self.mask.size)) * dy
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

