
# coding: utf-8

# In[2]:

# Authors: Daichi Yoshikawa <daichi.yoshikawa@gmail.com>
# License: BSD 3 clause

from __future__ import absolute_import

import numpy as np

from .layer import Layer
from ..utils import is_multi_channels_image
from ..utils import flatten, unflatten

class BatchNormLayer(Layer):
    """Implementation of Batch Normalization.

    Derived class of Layer.
    It doesn't calculate average and variance of all training data.
    Instead, it approximately get them through
    lowpass filter while training model.

    Parameters
    ----------
    gamma : np.array
        1d matrix which is used to scale the normalized value.
        (x'' = gamma * x' + beta)
        This value is updated while training
    beta : np.array
        1d matrix which is used to shift the normalized value.
        (x'' = gamma * x' + beta)
        This value is updated while training.
    miu : np.array
        1d matrix which is composed of feature-wise means.
    var : np.array
        1d matrix which is composed of feature-wise variances.
    momentum : float, default 0.9
        Intensity of lowpass filter used to calculate average miu and var.
    ep : float, default 1e-5
        Used to avoid 0 division.
    dtype : type
        Data type of all numeric values.

    Reference
    ---------
    Batch Normalization: Accelerating Deep Network Training
    by Reducing Internal Covariate Shift
    http://proceedings.mlr.press/v37/ioffe15.pdf
    """
    def __init__(self, momentum=0.9):
        self.gamma = None
        self.beta = None
        self.miu = None
        self.var = None
        self.momentum = momentum
        self.ep = 1e-5

    def set_dtype(self, dtype):
        """Set data type to use.

        Warning
        -------
        Not supposed to be called directly.
        Called automatically in the phase of NeuralNetwork's initialization.
        """
        self.dtype = dtype
        self.momentum = dtype(self.momentum)
        self.ep = dtype(self.ep)

    def get_type(self):
        return 'batch_norm'

    def set_parent(self, parent):
        Layer.set_parent(self, parent)
        self.output_shape = self.input_shape

    def forward(self, x):
        self.__forward(x)
        self.child.forward(self.fire)

    def backward(self, dy):
        self.__backward(dy)
        self.parent.backward(self.backfire)

    def predict_to_eval(self, x):
        self.__predict(x)
        return self.child.predict_to_eval(self.fire)

    def predict(self, x):
        self.__predict(x)
        return self.child.predict(self.fire)

    def finalize_training(self, x):
        self.__predict(x)
        self.child.finalize_training(self.fire)

    def __forward(self, x):
        if is_multi_channels_image(self.input_shape):
            x = flatten(x, self.input_shape)

        miu = np.mean(x, axis=0)
        self.xmiu = x - miu

        var = np.mean(self.xmiu**2, axis=0)
        self.std_inv = 1. / (np.sqrt(var + self.ep))

        if self.gamma is None:
            self.gamma = np.ones(self.input_shape, dtype=self.dtype)
        if self.beta is None:
            self.beta = np.zeros(self.input_shape, dtype=self.dtype)

        self.xhat = self.xmiu * self.std_inv
        self.fire = self.gamma * self.xhat + self.beta

        if self.miu is None:
            self.miu = miu
        if self.var is None:
            self.var = var

        self.miu *= self.momentum
        self.miu += (1. - self.momentum) * miu
        self.var *= self.momentum
        self.var += (1. - self.momentum) * var

        if is_multi_channels_image(self.output_shape):
            self.fire = unflatten(self.fire, self.input_shape)

    def __backward(self, dy):
        if is_multi_channels_image(self.output_shape):
            dy = flatten(dy, self.input_shape)

        batch_size = dy.shape[0]

        dbeta = dy.sum(axis=0)
        dgamma = (self.xhat * dy).sum(axis=0)

        tmp1 = (self.gamma * self.xmiu * dy).sum(axis=0)
        tmp2 = -np.power(self.std_inv, 3) * tmp1 / batch_size
        tmp3 = self.xmiu * tmp2 + self.gamma * self.std_inv * dy
        tmp4 = tmp3.sum(axis=0)

        self.backfire = tmp3 - tmp4 / batch_size
        self.beta -= dbeta / batch_size
        self.gamma -= dgamma / batch_size

        if is_multi_channels_image(self.input_shape):
            self.backfire = unflatten(self.backfire, self.input_shape)

    def __predict(self, x):
        if is_multi_channels_image(self.input_shape):
            x = flatten(x, self.input_shape)

        self.fire = self.gamma * (x - self.miu) / np.sqrt(self.var + self.ep) + self.beta

        if is_multi_channels_image(self.output_shape):
            self.fire = unflatten(self.fire, self.input_shape)

