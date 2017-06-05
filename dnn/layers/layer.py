
# coding: utf-8

# In[2]:

# Authors: Daichi Yoshikawa <daichi.yoshikawa@gmail.com>
# License: BSD 3 clause

from __future__ import absolute_import

import numpy as np

class Layer:
    """Base class for layers.

    Parameters
    ----------
    dtype : type
        Data type to use.
    fire : np.array
        Result of forward calculation in this layer.
    backfire : np.array
        Result of backward calculation in this layer.
    parent : Derived class of Layer
        Parent layer of this layer.
    child : Derived class of Layer
        Child layer of this layer.

    Warning
    -------
    This class should not be used directly.
    Use derived classes instead.
    """
    dtype = np.float64

    def __init__(self):
        self.fire = None
        self.backfire = None

    def set_dtype(self, dtype):
        pass

    def get_type(self):
        raise NotImplementedError('Layer.get_type')

    def has_weight(self):
        return False

    def set_parent(self, parent):
        """Set parent layer to this layer and
        set this layer to parent layer's child layer.

        Arguments
        ---------
        parent : Derived class of Layer
            Any reasonable layer.
        """
        self.parent = parent
        parent.child = self

    def forward(self, x):
        """Forward calculation called in training phase.

        Arguments
        ---------
        x : np.array
            fire of parent layer in 2d array.
            If no parent layer, it would be normalized descriptive features.
        """
        raise NotImplementedError('Layer.forward')

    def backward(self, dy):
        """Backward calculation.

        Arguments
        ---------
        dy : np.array
            backfire of child layer in 2d array.
            If no child layer, it would be errors of
            current predicted results against training data.
        """
        raise NotImplementedError('Layer.backward')

    def predict_to_eval(self, x):
        """Forward calculation called in evaluation phase.

        Arguments
        ---------
        x : np.array
            fire of parent layer in 2d array.
            If no parent layer, it would be normalized descriptive features.
        """
        raise NotImplementedError('Layer.predict_to_eval')

    def predict(self, x):
        """Forward calculation called in prediction phase.

        Arguments
        ---------
        x : np.array
            fire of parent layer in 2d array.
            If no parent layer, it would be normalized descriptive features.
        """
        raise NotImplementedError('Layer.predict')

    def finalize_training(self, x):
        """Implements finalizing training of layer.

        Arguments
        ---------
        x : np.array
            fire of parent layer in 2d array.
            If no parent layer, it would be normalized descriptive features.
        """
        raise NotImplementedError('Layer.finalize')


class InputLayer(Layer):
    """Implement the first layer of neural network.

    Derived class of Layer
    shape : tuple
        Shape of this layer's neurons.
        Eg. If you have descriptive feature matrix in form of
        (num of data, num of feature) such as
        [[x1-1, x1-2, ... , x1-100],
         [x2-1, x2-2, ... , x2-100],
         ...
         [xm-1, xm-2, ... , xm-100],
        ]
        Shape will be (100).
    """
    def __init__(self, shape):
        self.shape = shape

    def get_type(self):
        return 'input'

    def forward(self, x):
        """Starting point of forward calculation."""
        self.child.forward(x)

    def backward(self, dy):
        pass

    def predict_to_eval(self, x):
        return self.child.predict_to_eval(x)

    def predict(self, x):
        return self.child.predict(x)

    def finalize_training(self, x):
        self.child.finalize_training(x)


class OutputLayer(Layer):
    """Implements the last layer of neural network.

    Derived class of Layer.

    Parameters
    ----------
    shape : tuple
        Shape of this layer's neurons.
    """
    def __init__(self, shape):
        self.shape = shape

    def get_type(self):
        return 'output'

    def forward(self, x):
        self.fire = x

    def backward(self, dy):
        """Starting point of backward calculation."""
        self.backfire = dy
        self.parent.backward(self.backfire)

    def predict_to_eval(self, x):
        self.fire = x
        return self.fire

    def predict(self, x):
        self.fire = x
        return self.fire

    def finalize_training(self, x):
        self.fire = x

