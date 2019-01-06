# Authors: Daichi Yoshikawa <daichi.yoshikawa@gmail.com>
# License: BSD 3 clause

import numpy as np
from enum import Enum


class Optimizer:
    """
    Base class for optimizers.

    Warning: This class should not be used directly.
    Use derived classes instead.
    """
    Type = Enum(
            'Type',
            'sgd, momentum, ada_grad, adam, ada_delta, rms_prop, smorms3'
    )

    def get_type(self):
        raise NotImplementedError('Optimizer.get_type')

    def optimize(self, w, dw):
        raise NotImplementedError('Optimizer.update')

    def regularization(self, learning_rate, weight_decay):
        """
        Returns value which is supposed to multiplied to weights of
        neural network to keep them small values as possible.

        Attributes
        ----------
        learning_rate : float
            Learning rate. Generally > 0.0 and <= 0.3.
        weight_decay : float
            Degree of weight decay. Generally >= 0.0 and <= 0.3.

        Returns:
        float
            Degree of regularization.
        """

        return (1. - learning_rate * weight_decay)


class SGD(Optimizer):
    def __init__(self, dtype=np.float32, **kwargs):
        self.learning_rate = kwargs.pop('learning_rate', 3e-2)
        self.weight_decay = kwargs.pop('weight_decay', 0.)

    def get_type(self):
        return 'sgd'

    def optimize(self, w, dw):
        w[1:, :] *= self.regularization(self.learning_rate, self.weight_decay)
        w -= self.learning_rate * dw


class Momentum(Optimizer):
    def __init__(self, dtype=np.float32, **kwargs):
        self.dtype = dtype
        self.learning_rate = kwargs.pop('learning_rate', 3e-2)
        self.weight_decay = kwargs.pop('weight_decay', 0.)
        self.momentum_rate = kwargs.pop('momentum_rate', 0.9)
        self.pre_dw = None

    def get_type(self):
        return 'momentum'

    def optimize(self, w, dw):
        w[1:, :] *= self.regularization(self.learning_rate, self.weight_decay)

        if self.pre_dw is None:
            self.pre_dw = np.zeros_like(dw)

        self.pre_dw = self.learning_rate*dw + self.momentum_rate*self.pre_dw
        w -= self.pre_dw


class AdaGrad(Optimizer):
    """
    References
    ----------
    Adaptive Subgradient Methods for
    Online Learning and Stochastic Optimization
    http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf
    """
    def __init__(self, dtype=np.float32, **kwargs):
        self.dtype = dtype
        self.learning_rate = kwargs.pop('learning_rate', 3e-2)
        self.weight_decay = kwargs.pop('weight_decay', 0.)
        self.ep = kwargs.pop('ep', 1e-5)
        self.h = None

    def get_type(self):
        return 'ada_grad'

    def optimize(self, w, dw):
        w[1:, :] *= self.regularization(self.learning_rate, self.weight_decay)

        if self.h is None:
            self.h = np.zeros_like(w)

        self.h += np.power(dw, 2)
        dw *= self.dtype(1.) / np.sqrt(self.h + self.ep)
        w -= self.learning_rate * dw


class Adam(Optimizer):
    """
    References
    ----------
    Adam: A Method for Stochastic Optimization
    https://arxiv.org/pdf/1412.6980v9.pdf
    """
    def __init__(self, dtype=np.float32, **kwargs):
        self.dtype = dtype
        self.learning_rate = kwargs.pop('learning_rate', 1e-3)
        self.weight_decay = kwargs.pop('weight_decay', 0.)
        self.beta = kwargs.pop('beta', 0.9)
        self.gamma = kwargs.pop('gamma', 0.999)
        self.ep = kwargs.pop('ep', 1e-5)
        self.v = None
        self.r = None

    def get_type(self):
        return 'adam'

    def optimize(self, w, dw):
        w[1:, :] *= self.regularization(self.learning_rate, self.weight_decay)

        if self.v is None:
            self.v = np.zeros_like(w)

        if self.r is None:
            self.r = np.zeros_like(w)

        dw_square = np.power(dw, 2)
        self.v = self.beta * (self.v - dw) + dw
        self.r = self.gamma * (self.r - dw_square) + dw_square

        dw = self.learning_rate / np.sqrt(self.r + self.ep) * self.v
        w -= dw


class AdaDelta(Optimizer):
    """
    References
    ADADELTA: AN ADAPTIVE LEARNING RATE METHOD
    http://www.matthewzeiler.com/pubs/googleTR2012/googleTR2012.pdf
    """
    def __init__(self, dtype=np.float32, **kwargs):
        self.dtype = dtype
        self.learning_rate = kwargs.pop('learning_rate', 1e-3)
        self.weight_decay = kwargs.pop('weight_decay', 0.)
        self.gamma = kwargs.pop('gamma', 0.95)
        self.ep = kwargs.pop('ep', 1e-5)
        self.r = None
        self.s = None
        self.v = None

    def optimize(self, w, dw):
        w[1:, :] *= self.regularization(self.learning_rate, self.weight_decay)

        if self.r is None:
            self.r = np.zeros_like(w)
        if self.s is None:
            self.s = np.zeros_like(w)
        if self.v is None:
            self.v = np.zeros_like(w)

        self.r = self.gamma * self.r + (1. - self.gamma) * np.power(dw, 2)
        self.v = np.sqrt(self.s + self.ep) / (np.sqrt(self.r + self.ep)) * dw
        w -= self.learning_rate * self.v
        self.s = self.gamma + (1. - self.gamma) * np.power(self.v, 2)


class RMSProp(Optimizer):
    """
    References
    ----------
    Neural Networks for Machine Learning
    Lecture 6a
    Overview of mini-batch gradient descent
    http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
    """
    def __init__(self, dtype=np.float32, **kwargs):
        self.dtype = dtype
        self.learning_rate = kwargs.pop('learning_rate', 1e-3)
        self.weight_decay = kwargs.pop('weight_decay', 0.)
        self.gamma = kwargs.pop('gamma', 0.9)
        self.ep = kwargs.pop('ep', 1e-5)
        self.h = None

    def optimize(self, w, dw):
        w[1:, :] *= self.regularization(self.learning_rate, self.weight_decay)

        if self.h is None:
            self.h = np.zeros_like(w)

        self.h = self.gamma * self.h + (1. - self.gamma) * np.power(dw, 2)
        dw *= 1. / (np.sqrt(self.h) + self.ep)
        w -= self.learning_rate * dw


class SMORMS3(Optimizer):
    """
    References
    ----------
    Rmsprop loses to smorms3 - beware the epsilon!
     http://sifter.org/˜simon/journal/20150420.html
    """
    def __init__(self, dtype=np.float32, **kwargs):
        self.dtype = dtype
        self.learning_rate = kwargs.pop('learning_rate', 1e-3)
        self.weight_decay = kwargs.pop('weight_decay', 0.)
        self.ep = kwargs.pop('ep', 1e-5)
        self.s = 1.
        self.v = None
        self.r = None
        self.x = None

    def optimize(self, w, dw):
        w[1:, :] *= self.regularization(self.learning_rate, self.weight_decay)

        if self.v is None:
            self.v = np.zeros_like(w)
        if self.r is None:
            self.r = np.zeros_like(w)
        if self.x is None:
            self.x = np.zeros_like(w)

        beta = 1. / self.s

        self.v = beta * self.v + (1. - beta) * dw
        self.r = beta * self.r + (1. - beta) * np.power(dw, 2)
        self.x = np.power(self.v, 2) / (self.r + self.ep)

        dw *= np.minimum(self.x, self.learning_rate)
        dw /= np.sqrt(self.r) + self.ep
        w -= dw
        self.s = 1. + (1. - self.x) * self.s
