
# coding: utf-8

# In[10]:

# Authors: Daichi Yoshikawa <daichi.yoshikawa@gmail.com>
# License: BSD 3 clause

from __future__ import absolute_import

import numpy as np
from enum import Enum

from .layer import Layer

class ActivationLayer(Layer):
    """Implements layer which convert values by activation function.

    Parameters
    ----------
    activation : Derived class of Activation
        Activation function to use.
    shape : tuple
        Shape of this layer's neurons.
    """
    def __init__(self, activation):
        """
        Arguments
        ---------
        activation : Activation.Type
           Name of activation function to use.
        """
        self.activation = ActivationFactory.get(activation)

    def get_type(self):
        return 'activation'

    def set_parent(self, parent):
        Layer.set_parent(self, parent)
        self.shape = parent.shape

    def forward(self, x):
        self.fire = self.activation.activate(x)
        self.child.forward(self.fire)

    def backward(self, dy):
        self.backfire = dy * self.activation.grad(self.fire)
        self.parent.backward(self.backfire)

    def predict_to_eval(self, x):
        self.fire = self.activation.activate(x)
        return self.child.predict_to_eval(self.fire)

    def predict(self, x):
        self.fire = self.activation.activate(x)
        return self.child.predict(self.fire)

    def finalize_training(self, x):
        self.fire = self.activation.activate(x)
        self.child.finalize_training(self.fire)

# Implement Activation Functions
class Activation:
    """Base class for activation functions.

    Warning
    -------
    This class should not be used directly.
    Use derived classes instead. 
    """
    Type = Enum('Type', 'sigmoid, relu, tanh, softmax')

    def get_type(self):
        """Interface to get type of activation in string."""
        raise NotImplementedError('Activation.get_type')

    def activate(self, x):
        """Interface to get f(x), where f(x) is activation function."""
        raise NotImplementedError('Activation.activate')

    def grad(self, x):
        """Interface to get df(x)/dx, where f(x) is activation function."""
        raise NotImplementedError('Activation.grad')


class Sigmoid(Activation):
    """Sigmoid function.

    f(x) = 1 / (1 + exp(-a*x)).
    df(x)/dx = (1 - f(x)) * f(x)
    This class implements aboves with a == 1.
    """
    def get_type(self):
        return 'sigmoid'

    def activate(self, x):
        return 1. / (1. + np.exp(-x))

    def grad(self, x):
        return (1. - x) * x


class ReLU(Activation):
    """Rectified linear function.

    f(x) = x when x > 0 and f(x) = 0 when x <= 0.
    df(x)/dx = 1 when x > 0 and df(x)/dx = 0 when x <= 0.

    Parameters
    ----------
    __mask : np.array
        2d matrix whose entry(i, j) is 
        1 when x(i, j) > 0 and 0 when x(i, j) == 0.
    """
    def get_type(self):
        return 'relu'

    def activate(self, x):
        self.__mask = (x > 0.).astype(x.dtype)
        return x * self.__mask

    def grad(self, x):
        return self.__mask


class Tanh(Activation):
    """Tanh function.

    f(x) = tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    df(x)/dx = 1 - f(x)^2
    """
    def get_type(self):
        return 'tanh'

    def activate(self, x):
        return np.tanh(x)

    def grad(self, x):
        return 1. - np.power(x, 2, dtype=x.dtype)


class Softmax(Activation):
    """Softmax function to convert values into probabilities.

    f(x) = exp(x) / exp(x).sum()
         = exp(x) * exp(-c) / (exp(x).sum()) * exp(-c)
         = exp(x - c) / exp(x - c).sum()
    where c == x.max(), to avoid overflow of exp calculation.
    """
    def get_type(self):
        return 'softmax'

    def activate(self, x):
        var = np.exp(x - x.max())
        return var / var.sum(axis=1).reshape(var.shape[0], 1)

    def grad(self, x):
        return 1.


class ActivationFactory:
    """Factory class to get activation's instance.

    Warning
    -------
    Get activation's instance through this class.
    """
    __activation = {
            Activation.Type.sigmoid : Sigmoid(),
            Activation.Type.relu : ReLU(),
            Activation.Type.tanh : Tanh(),
            Activation.Type.softmax : Softmax(),
    }

    @classmethod
    def get(cls, activation):
        """Returns instance of selected activation function.

        Arguments
        ---------
        activation : Activation.Type
            Name of activation function to use.

        Returns
        -------
        Derived class of Activation
            Instance of selected activation function.
        """
        return cls.__activation[activation]

