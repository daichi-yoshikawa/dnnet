# Authors: Daichi Yoshikawa <daichi.yoshikawa@gmail.com>
# License: BSD 3 clause

import dnnet.utils.numcupy as ncp
from dnnet.ext_mathlibs import cp, np
from dnnet.layers.layer import Layer
from dnnet.utils.nn_utils import is_multi_channels_image
from dnnet.utils.nn_utils import asnumpy, flatten, unflatten


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
    def __init__(self, momentum=0.9, force_cpu=False):
        self.gamma = None
        self.beta = None
        self.miu = None
        self.var = None
        self.momentum = momentum
        self.ep = 1e-5
        self.force_cpu = force_cpu

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

    def get_config_str_tail(self):
        return ''

    def set_parent(self, parent):
        Layer.set_parent(self, parent)
        self.output_shape = self.input_shape

    def forward(self, x):
        self.__forward(x)
        self.child.forward(self.fire)

    def backward(self, dy):
        self.__backward(dy)
        self.parent.backward(self.backfire)

    def predict(self, x):
        self.__predict(x)
        return self.child.predict(self.fire)

    def __forward(self, x):
        x = x if self.force_cpu else cp.array(x)
        miu = ncp.mean(x, axis=0)
        xmiu = x - miu

        var = ncp.mean(xmiu**2, axis=0)
        std_inv = 1. / (ncp.sqrt(var + self.ep))

        gamma, beta = None, None
        shape = self.input_shape
        if self.gamma is None:
            gamma = ncp.ones(shape, dtype=self.dtype, arr_type=type(x))
        else:
            gamma = self.gamma if self.force_cpu else cp.array(self.gamma)
        if self.beta is None:
            beta = ncp.zeros(shape, dtype=self.dtype, arr_type=type(x))
        else:
            beta = self.beta if self.force_cpu else cp.array(self.beta)

        xhat = xmiu * std_inv
        fire = gamma * xhat + beta

        pre_miu, pre_var = None, None
        if self.miu is None:
            pre_miu = miu
        else:
            pre_miu = self.miu if self.force_cpu else cp.array(self.miu)
        if self.var is None:
            pre_var = var
        else:
            pre_var = self.var if self.force_cpu else cp.array(self.var)
        miu = pre_miu * self.momentum + (1. - self.momentum) * miu
        var = pre_var * self.momentum + (1. - self.momentum) * var

        self.xmiu = asnumpy(xmiu)
        self.var = asnumpy(var)
        self.std_inv = asnumpy(std_inv)
        self.gamma = asnumpy(gamma)
        self.beta = asnumpy(beta)
        self.xhat = asnumpy(xhat)
        self.fire = asnumpy(fire)
        self.miu = asnumpy(miu)
        self.var = asnumpy(var)

    def __backward(self, dy):
        dy = dy if self.force_cpu else cp.array(dy)
        xhat = self.xhat if self.force_cpu else cp.array(self.xhat)
        xmiu = self.xmiu if self.force_cpu else cp.array(self.xmiu)
        std_inv = self.std_inv if self.force_cpu else cp.array(self.std_inv)
        beta = self.beta if self.force_cpu else cp.array(self.beta)
        gamma = self.gamma if self.force_cpu else cp.array(self.gamma)

        batch_size = dy.shape[0]
        dbeta = dy.sum(axis=0)
        dgamma = (xhat * dy).sum(axis=0)

        tmp1 = (gamma * xmiu * dy).sum(axis=0)
        tmp2 = -ncp.power(std_inv, 3) * tmp1 / batch_size
        tmp3 = xmiu*tmp2 + gamma*std_inv*dy
        tmp4 = tmp3.sum(axis=0)

        backfire = tmp3 - tmp4/batch_size
        beta = beta - dbeta/batch_size
        gamma = gamma - dgamma/batch_size

        self.backfire = asnumpy(backfire)
        self.beta = asnumpy(beta)
        self.gamma = asnumpy(gamma)

    def __predict(self, x):
        x = x if self.force_cpu else cp.array(x)
        gamma = self.gamma if self.force_cpu else cp.array(self.gamma)
        beta = self.beta if self.force_cpu else cp.array(self.beta)
        miu = self.miu if self.force_cpu else cp.array(self.miu)
        var = self.var if self.force_cpu else cp.array(self.var)

        fire = gamma * (x - miu) / ncp.sqrt(var + self.ep) + beta

        self.fire = asnumpy(fire)
