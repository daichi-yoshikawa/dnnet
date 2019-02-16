import sys
import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict


class LearningCurve:
    """Class to manage historical losses and accuracies.

    Used to record historical losses and accuracies for train/test data.
    Recorded data is printed out them while training model and also plotted
    through method.

    vals : OrderedDict of np.array
        Key is type of data to record and value is np.array
        which records target data in 1d array.
    label : OrderedDict of str
        Key is type of data to record and value is label which is used
        when printing out it.
    digits : OrderedDict of str
        Key is type of data to record and value is str which defines
        number of digits to print out.
    dtype : type
        Data type of values of vals.
    """
    def __init__(self, dtype=np.float32):
        self.vals = OrderedDict((
            ('loss_train', None),
            ('acc_train', None),
            ('loss_test', None),
            ('acc_test', None),
        ))
        self.label = OrderedDict((
            ('loss_train', 'loss'),
            ('acc_train', 'acc'),
            ('loss_test', 'loss(test)'),
            ('acc_test', 'acc(test)'),
        ))
        self.digits = OrderedDict((
            ('loss_train', '%2.5f'),
            ('acc_train', '%3.3f'),
            ('loss_test', '%2.5f'),
            ('acc_test', '%3.3f'),
        ))
        self.dtype = dtype

    def get_loss(self):
        """Returns losses w.r.t training and test data."""
        return self.vals['loss_train'], self.vals['loss_test']

    def get_acc(self):
        """Returns accuracies w.r.t training and test data."""
        return self.vals['acc_train'], self.vals['acc_test']

    def get(self):
        """Returns losses and accuracies w.r.t training and test data."""
        return (self.vals['loss_train'], self.vals['acc_train'],
                self.vals['loss_test'], self.vals['acc_test'])

    def add(self, loss_train, loss_test=None, acc_train=None, acc_test=None):
        """Records losses and accuracies for training and test data.

        Arguments
        ---------
        loss_train : float
            Loss value w.r.t training data.
        loss_test : float, default None
            Loss value w.r.t test data. If None, this value is ignored.
        acc_train : float, default None
            Accuracy w.r.t training data.
            In the case of regression, this will be None.
            If None, this value is ignored.
        acc_test : float, default None
            Accuracy w.r.t test data.
            In the case of regression, this will be None.
            If None, this value is ignored.
        """
        self.__add('loss_train', loss_train)
        self.__add('loss_test', loss_test)
        self.__add('acc_train', acc_train)
        self.__add('acc_test', acc_test)

    def stdout(self, epoch):
        """Prints out losses and accuracies which are recorded at last.

        Arguments
        ---------
        epoch : int
            Number of epoch.
        """
        sys.stdout.write('epoch:%2d' % (epoch+1))

        for key, val in self.vals.items():
            self.__stdout(key, val)

        sys.stdout.write('\n')

################################################################################
    def plot(
            self, loss_range=None, acc_range=None, figsize=(8, 10),
            fontsize=12):
        """Plots curves of losses and accuracies w.r.t training and test data.

        Arguments
        ---------
        loss_range : tuple ((xmin, xmax), (ymin, ymax)), default None
            Range of x and y for figure of loss curves.
            x and y correspond with number of epoch and loss respectively.
            If None, range of x and y are not set explicitly.
        acc_range : tuple ((xmin, xmax), (ymin, ymax)), default None
            Range of x and y for figure of accuracy curves.
            x and y correspond with number of epoch and accuracy respectively.
            If None, range of x and y are not set explicitly.
        figsize : tuple (x, y), default (8, 10)
            Size of figure.
        fontsize : int
            Size of font which is used for labels of figure.
        """
        epochs = 0

        for key, val in self.vals.items():
            if val is not None:
                epochs = np.arange(val.size)
                break

        if epochs.size == 0:
            msg = 'LearningCurve.plot failed. No data is recorded.'
            raise RuntimeError(msg)

        if not self.__recorded_loss_or_acc():
            sys.stdout.write('No recorded loss and accuracy.\n')
            return

        plt.figure(figsize=figsize)

        recorded_loss_and_acc = self.__recorded_loss_and_acc()
        if recorded_loss_and_acc:
            plt.subplot(2, 1, 1)
        if self.vals['loss_train'] is not None:
            plt.plot(epochs, self.vals['loss_train'])
        if self.vals['loss_test'] is not None:
            plt.plot(epochs, self.vals['loss_test'])
        if loss_range is not None:
            xrange, yrange = loss_range
            xmin, xmax = xrange
            ymin, ymax = yrange
            plt.xlim(xmin, xmax)
            plt.ylim(ymin, ymax)
        if self.__recorded_loss():
            plt.xlabel('Epochs[-]', fontsize=fontsize, fontname='serif')
            plt.ylabel('Loss', fontsize=fontsize, fontname='serif')
            plt.title('Loss')

        if recorded_loss_and_acc:
            plt.subplot(2, 1, 2)
        if self.vals['acc_train'] is not None:
            plt.plot(epochs, self.vals['acc_train'])
        if self.vals['acc_test'] is not None:
            plt.plot(epochs, self.vals['acc_test'])
        if loss_range is not None:
            xrange, yrange = acc_range
            xmin, xmax = xrange
            ymin, ymax = yrange
            plt.xlim(xmin, xmax)
            plt.ylim(ymin, ymax)
        if self.__recorded_acc():
            plt.xlabel('Epochs[-]', fontsize=fontsize, fontname='serif')
            plt.ylabel('Accuracy', fontsize=fontsize, fontname='serif')
            plt.title('Accuracy')

        if recorded_loss_and_acc:
            plt.tight_layout()

        plt.show()

    def __add(self, key, val):
        """Record value selected by key."""
        if val is None:
            return

        if self.vals[key] is None:
            self.vals[key] = np.array([], dtype=self.dtype)
        self.vals[key] = np.append(self.vals[key], val)

    def __stdout(self, key, val):
        """Print out selected value."""
        if val is None:
            return

        msg = ' ' + self.label[key] + ': ' + self.digits[key]
        sys.stdout.write(msg % (self.vals[key][-1]))

    def __recorded_loss(self):
        """Returns true if at least one loss has been recorded."""
        loss_train_is_none = (self.vals['loss_train'] is None)
        loss_test_is_none = (self.vals['loss_test'] is None)

        if loss_train_is_none and loss_test_is_none:
            return False
        return True

    def __recorded_acc(self):
        """Returns true if at least one accuracy has been recorded."""
        acc_train_is_none = self.vals['acc_train'] is None
        acc_test_is_none = self.vals['acc_test'] is None
        if acc_train_is_none and acc_test_is_none:
            return False
        return True

    def __recorded_loss_and_acc(self):
        """Returns true if at least a pair of loss and accuracy data
        has been already recorded.
        """
        if self.__recorded_loss() and self.__recorded_acc():
            return True
        return False

    def __recorded_loss_or_acc(self):
        """Returns true if at least one loss data or accuracy data
        has been already recorded.
        """
        if self.__recorded_loss() or self.__recorded_acc():
            return True
        return False
