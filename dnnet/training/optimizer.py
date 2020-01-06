# Authors: Daichi Yoshikawa <daichi.yoshikawa@gmail.com>
# License: BSD 3 clause

from enum import Enum

from dnnet.ext_mathlibs import cp, np
from dnnet.utils.nn_utils import asnumpy


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
        w_ = cp.array(w)
        dw_ = cp.array(dw)

        w_[1:, :] *= self.regularization(self.learning_rate, self.weight_decay)
        w_ -= self.learning_rate * dw_

        w[::] = asnumpy(w_)


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
        w_ = cp.array(w)
        dw_ = cp.array(dw)

        w_[1:, :] *= self.regularization(self.learning_rate, self.weight_decay)

        pre_dw = cp.zeros_like(dw_) if self.pre_dw is None else cp.array(self.pre_dw)
        pre_dw = self.learning_rate*dw_ + self.momentum_rate*pre_dw

        w_ -= pre_dw

        w[::] = asnumpy(w_)
        self.pre_dw = asnumpy(pre_dw)


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
        w_ = cp.array(w)
        dw_ = cp.array(dw)

        w_[1:, :] *= self.regularization(self.learning_rate, self.weight_decay)
        h = cp.zeros_like(w_) if self.h is None else cp.array(self.h)

        h += cp.power(dw_, 2)
        w_ -= self.learning_rate * (self.dtype(1.) / cp.sqrt(h + self.ep)) * dw_

        w[::] = asnumpy(w_)
        self.h = h


class Adam(Optimizer):
    """
    Warnings
    --------
    If Overflow exception is raised, try increasing beta.

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
        w_ = cp.array(w)
        dw_ = cp.array(dw)

        w_[1:, :] *= self.regularization(self.learning_rate, self.weight_decay)
        v = cp.zeros_like(w_) if self.v is None else cp.array(self.v)
        r = cp.zeros_like(w_) if self.r is None else cp.array(self.r)

        dw_square = cp.power(dw_, 2)
        v = self.beta * (v - dw_) + dw_
        r = self.gamma * (r - dw_square) + dw_square

        w_ -= self.learning_rate / cp.sqrt(r + self.ep) * v

        w[::] = asnumpy(w_)
        self.v = asnumpy(v)
        self.r = asnumpy(r)


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
        w_ = cp.array(w)
        dw_ = cp.array(dw)

        w_[1:, :] *= self.regularization(self.learning_rate, self.weight_decay)

        r = cp.zeros_like(w_) if self.r is None else cp.array(self.r)
        s = cp.zeros_like(w_) if self.s is None else cp.array(self.s)
        v = cp.zeros_like(w_) if self.v is None else cp.array(self.v)

        r = self.gamma * r + (1. - self.gamma) * cp.power(dw_, 2)
        v = cp.sqrt(s + self.ep) / (cp.sqrt(r + self.ep)) * dw_

        w_ -= self.learning_rate * v
        s = self.gamma + (1. - self.gamma) * cp.power(v, 2)

        w[::] = asnumpy(w_)
        self.r = asnumpy(r)
        self.s = asnumpy(s)
        self.v = asnumpy(v)


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
        self.gamma = kwargs.pop('gamma', 0.99)
        self.ep = kwargs.pop('ep', 1e-5)
        self.h = None

    def optimize(self, w, dw):
        w_ = cp.array(w)
        dw_ = cp.array(dw)

        w_[1:, :] *= self.regularization(self.learning_rate, self.weight_decay)
        h = cp.zeros_like(w_) if self.h is None else cp.array(self.h)
        h = self.gamma * h + (1. - self.gamma) * cp.power(dw_, 2)
        w_ -= self.learning_rate * dw_ / (cp.sqrt(h) + self.ep)

        w[::] = asnumpy(w_)
        self.h = asnumpy(h)


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
