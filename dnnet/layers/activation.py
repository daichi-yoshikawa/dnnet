# Authors: Daichi Yoshikawa <daichi.yoshikawa@gmail.com>
# License: BSD 3 clause

from enum import Enum

import dnnet.utils.numcupy as ncp
from dnnet.ext_mathlibs import cp, np
from dnnet.layers.layer import Layer
from dnnet.utils.nn_utils import is_multi_channels_image
from dnnet.utils.nn_utils import asnumpy, flatten, unflatten


class ActivationLayer(Layer):
    """Implements layer which convert values by activation function.

    Parameters
    ----------
    activation : Derived class of Activation
        Activation function to use.
    """
    def __init__(self, activation, force_cpu=False):
        """
        Arguments
        ---------
        activation : Activation.Type
           Name of activation function to use.
        """
        self.layer_index = 0
        self.activation = ActivationFactory.get(activation, force_cpu)

    def get_type(self):
        return 'activation'

    def get_config_str_tail(self):
        tail = 'func: %s' % self.activation.get_type()
        return tail

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
        self.__forward(x)
        return self.child.predict(self.fire)

    def __forward(self, x):
        self.fire = self.activation.activate(x)

    def __backward(self, dy):
        self.backfire = dy * self.activation.grad(self.fire)


# Implement Activation Functions
class Activation:
    """Base class for activation functions.

    Warning
    -------
    This class should not be used directly.
    Use derived classes instead.
    """
    Type = Enum('Type', 'sigmoid, relu, elu, srrelu, tanh, softmax')

    def __init__(self, force_cpu):
        self.force_cpu = force_cpu

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
    def __init__(self, force_cpu):
        super().__init__(force_cpu=force_cpu)

    def get_type(self):
        return 'sigmoid'

    def activate(self, x):
        return 1. / (1. + ncp.exp(-x))

    def grad(self, x):
        x_ = x if self.force_cpu else cp.array(x)
        return asnumpy((1. - x_) * x_)


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
    def __init__(self, force_cpu):
        super().__init__(force_cpu=force_cpu)

    def get_type(self):
        return 'relu'

    def activate(self, x):
        x_ = x if self.force_cpu else cp.array(x)
        mask = (x_ > 0.0).astype(x_.dtype)
        self.__mask = asnumpy(mask)
        return asnumpy(x_ * mask)

    def grad(self, x):
        return self.__mask


class ELU(Activation):
    """Exponential Linear Units.

    Parameters
    ----------
    alpha : float, default 1.0
        Controls the value to which an ELU saturates for negative net inputs.
    __mask : np.array
        2d matrix whose entry(i, j) is
        1 when x(i, j) > 0 and 0 when x(i, j) == 0.

    References
    ----------
    Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs)
    https://arxiv.org/pdf/1511.07289.pdf
    """
    def __init__(self, force_cpu):
        super().__init__(force_cpu=force_cpu)

    def get_type(self):
        return 'elu'

    def activate(self, x):
        self.__mask = (x > 0.0).astype(x.dtype)
        return self.__mask * x + ~(self.__mask) * (np.exp(x) - 1)

    def grad(self, x):
        return self.__mask + ~(self.__mask) * (x + 1)


class SRReLU(Activation):
    """Square Root version of Rectified Linear.

    To reduce bias shift and enhance nonlinearity, use square root of input
    instead of original value.

    Parameters
    ----------
    __mask : np.array
        2d matrix whose entry(i, j) is
        1 when x(i, j) > 0 and 0 when x(i, j) == 0.
    """
    def __init__(self, force_cpu):
        super().__init__(force_cpu=force_cpu)

    def get_type(self):
        return 'selu'

    def activate(self, x):
        self.__mask = (x > 0.0).astype(x.dtype)
        return (np.sqrt(self.__mask * x + 1) - 1)

    def grad(self, x):
        return self.__mask * 0.5 / (x + 1)


class Tanh(Activation):
    """Tanh function.

    f(x) = tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    df(x)/dx = 1 - f(x)^2
    """
    def __init__(self, force_cpu):
        super().__init__(force_cpu=force_cpu)

    def get_type(self):
        return 'tanh'

    def activate(self, x):
        return np.tanh(x).astype(x.dtype)

    def grad(self, x):
        return 1. - np.power(x, 2, dtype=x.dtype)


class Softmax(Activation):
    def __init__(self, force_cpu):
        super().__init__(force_cpu=force_cpu)

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
    @classmethod
    def get(cls, activation, force_cpu):
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
        if activation is Activation.Type.sigmoid:
            return Sigmoid(force_cpu=force_cpu)
        elif activation is Activation.Type.relu:
            return ReLU(force_cpu=force_cpu)
        elif activation is Activation.Type.elu:
            return ELU(force_cpu=force_cpu)
        elif activation is Activation.Type.srrelu:
            return SRReLU(force_cpu=force_cpu)
        elif activation is Activation.Type.tanh:
            return Tanh(force_cpu=force_cpu)
        elif activation is Activation.Type.softmax:
            return Softmax(force_cpu=force_cpu)
