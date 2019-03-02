# Authors: Daichi Yoshikawa <daichi.yoshikawa@gmail.com>
# License: BSD 3 clause

import numpy as np
from enum import Enum


class LossFunction:
    """Base class for loss functions.

    Warning
    -------
    This class should not be used directly.
    Use derived classes instead.

    Parameters
    ----------
    Type : Enum
        Enumeration of type of loss functions.
    ep : float
        Used to avoid log(0) computation.
    """
    Type = Enum(
        'Type',
        'multinomial_cross_entropy, binomial_cross_entropy, squared_error'
    )
    ep = 1e-5

    def get(self, y, t):
        """Returns loss.

        Arguments
        ---------
        y : np.array
            Predicted results in 2d array,
            whose shape is (num of data, num of predicted target features).
        t : np.array
            Target features of training data in 2d array,
            whose shape is (num of data, num of target features).

        Returns
        -------
        float
            Resulting loss value.
        """
        raise NotImplementedError('LossFunction.get')


class MultinomialCrossEntropy(LossFunction):
    """Loss function which is used for multi-class classification."""
    def get(self, y, t):
        return (-t * np.log(y + self.ep)).sum() / y.shape[0]


class BinomialCrossEntropy(LossFunction):
    """Loss function which is used for binary-class classification."""
    def get(self, y, t):
        error = -t * np.log(y + self.ep) - (1 - t) * np.log(1 - y + self.ep)
        return error.sum() / y.shape[0]


class SquaredError(LossFunction):
    """Loss function which is used for regression."""
    def get(self, y, t):
        return 0.5 * np.power(y - t, 2, dtype=y.dtype).sum() / y.shape[0]


class LossFunctionFactory:
    """Factory class to get loss function's instance.

    Warning
    -------
    Get loss function's instance through this class.
    """
    __loss_function = {
        LossFunction.Type.multinomial_cross_entropy:
        MultinomialCrossEntropy(),
        LossFunction.Type.binomial_cross_entropy:
        BinomialCrossEntropy(),
        LossFunction.Type.squared_error:
        SquaredError(),
    }

    @classmethod
    def get(cls, loss_function):
        """Returns instance of selected loss function.

        Arguments
        ---------
        loss_function : LossFunction.Type
            Name of loss function to use.

        Returns
        -------
        Derived class of LossFunction
            Instance of selected loss function.
        """
        return cls.__loss_function[loss_function]
