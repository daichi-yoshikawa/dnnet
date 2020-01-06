# Authors: Daichi Yoshikawa <daichi.yoshikawa@gmail.com>
# License: BSD 3 clause

from enum import Enum

from dnnet.ext_mathlibs import cp, np
from dnnet.utils.nn_utils import prod


class WeightInitialization:
    """Base class for random initialization of weight.

    Parameters
    ----------
    Type : Enum
        Enumeration of name of methods to generate random weight.
    """
    Type = Enum('Type', 'default, xavier, he')

    def get_type(self):
        """
        Returns
        -------
        str
            Name of method.
        """
        msg = 'WeightInitialization.get_type is called.'
        raise NotImplementedError(msg)

    def get(self, rows, cols, layer):
        """Returns weight set to random values based on selected method.

        Arguments
        ---------
        rows : int
            Number of rows of resulting weight.
        cols : int
            Number of cols of resulting weight.
        layer : Layer or its derived class
            Instance of layer which is associated with weights.

        Returns
        -------
        np.array
            Resulting weight in 2d array.
        """
        return self.get_var(layer) * np.random.randn(rows, cols)

    def get_var(self, layer):
        msg = 'WeightInitialization.get_var is called.'
        raise NotImplementedError(msg)


class DefaultInitialization(WeightInitialization):
    """Default method to randomly initialize weight.

    Reference
    ---------
    Coursera : Machine Learning
    """
    def get_type(self):
        return 'default'

    def get_var(self, layer):
        inout_size = prod(layer.input_shape) + prod(layer.output_shape)
        var = np.sqrt(6) / np.sqrt(inout_size)
        return var


class Xavier(WeightInitialization):
    """Implements Xavier's random initialization.

    Often used with sigmoid, tanh activation functions.

    Reference
    ---------
    Understanding the difficulty of training deep feedforward neural networks
    http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
    """
    def get_type(self):
        return 'xavier'

    def get_var(self, layer):
        in_size = prod(layer.input_shape)
        var = 1. / np.sqrt(in_size)
        return var


class He(WeightInitialization):
    """Implements He's random initialization.

    Often used with relu activation function.

    Reference
    ---------
    Delving Deep into Rectifiers:
    Surpassing Human-Level Performance on ImageNet Classification
    https://arxiv.org/pdf/1502.01852v1.pdf
    """
    def get_type(self):
        return 'he'

    def get_var(self, layer):
        in_size = prod(layer.input_shape)
        var = np.sqrt(2.) / np.sqrt(in_size)
        return var
