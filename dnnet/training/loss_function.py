# Authors: Daichi Yoshikawa <daichi.yoshikawa@gmail.com>
# License: BSD 3 clause

from dnnet.ext_mathlibs import cp, np


class LossFunction:
    """Base class for loss functions.

    Warning
    -------
    This class should not be used directly.
    Use derived classes instead.

    Parameters
    ----------
    ep : float
        Used to avoid log(0) computation.
    """
    ep = 1e-5

    def get_type(self):
        raise NotImplementedError('LossFunction.get_type is called.')

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
        raise NotImplementedError('LossFunction.get is called.')


class MultinomialCrossEntropy(LossFunction):
    """Loss function which is used for multi-class classification."""
    def get_type(self):
        return 'multinomial cross entropy'

    def get(self, y, t):
        return (-t * np.log(y + self.ep)).sum() / y.shape[0]


class BinomialCrossEntropy(LossFunction):
    """Loss function which is used for binary-class classification."""
    def get_type(self):
        return 'binomial cross entropy'

    def get(self, y, t):
        error = -t * np.log(y + self.ep) - (1 - t) * np.log(1 - y + self.ep)
        return error.sum() / y.shape[0]


class SquaredError(LossFunction):
    """Loss function which is used for regression."""
    def get_type(self):
        return 'squared error'

    def get(self, y, t):
        return 0.5 * np.power(y - t, 2, dtype=y.dtype).sum() / y.shape[0]
