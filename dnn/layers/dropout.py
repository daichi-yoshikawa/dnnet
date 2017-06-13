
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
        self.mask = np.arange(input_size).reshape(input_size)

    def forward(self, x):
        self.__forward(x)
        self.child.forward(self.fire)

    def backward(self, dy):
        self.__backward(dy)
        self.parent.backward(self.backfire)

    def predict(self, x):
        self.__predict(x)
        return self.child.predict(self.fire)

    def finalize_training(self, x):
        self.__predict(x)
        self.child.finalize_training(self.fire)

    def __forward(self, x):
        if is_multi_channels_image(self.input_shape):
            x = flatten(x, self.input_shape)

        np.random.shuffle(self.mask.reshape(self.mask.size))
        self.fire = (self.mask >= int(self.drop_ratio*self.mask.size)) * x

        if is_multi_channels_image(self.output_shape):
            self.fire = unflatten(self.fire, self.input_shape)

    def __backward(self, dy):
        if is_multi_channels_image(self.output_shape):
            dy = flatten(dy, self.input_shape)

        self.backfire = (self.mask >= int(self.drop_ratio*self.mask.size)) * dy

        if is_multi_channels_image(self.input_shape):
            self.backfire = unflatten(self.backfire, self.input_shape)

    def __predict(self, x):
        if is_multi_channels_image(self.input_shape):
            x = flatten(x, self.input_shape)        

        self.fire = (1. - self.drop_ratio) * x

        if is_multi_channels_image(self.output_shape):
            self.fire = unflatten(self.fire, self.input_shape)

