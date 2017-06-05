
# coding: utf-8

# In[1]:

# Authors: Daichi Yoshikawa <daichi.yoshikawa@gmail.com>
# License: BSD 3 clause

from __future__ import absolute_import

import numpy as np

class Monitor:
    """Implements feature to print out evaluation results.
    
    Warning
    -------
    Not supposed to be called directly.
    Called while training neural network model.
    """
    def print_loss(self, loss_train, loss_test, acc_train, acc_test, epoch):
        """Print losses and accuracies w.r.t training data and test data.

        If test data was not used, loss_test is None and
        losses and accuracies w.r.t test data are not printed.

        Arguments
        ---------
        loss_train : float
           Loss value w.r.t training data.
        loss_test : float or None
           Loss value w.r.t test data.
        acc_train : float
           Accuracy w.r.t training data.
        acc_test : float or None
           Accuracy w.r.t test data.
        epoch : int
           Number of epoch.
        """
        if loss_test is None:
            self.__print_loss_train(
                    loss_train, acc_train, epoch
            )
        else:
            self.__print_loss(
                    loss_train, loss_test, acc_train, acc_test, epoch
            )

    def __print_loss_train(self, loss_train, acc_train, epoch):
        if acc_train is None:
            print('epoch: %4d, loss: %2.6f' % (epoch + 1, loss_train))
        else:
            print('epoch: %4d, loss: %2.6f, acc: %3.3f'                  % (epoch + 1, loss_train, acc_train))
        
    def __print_loss(self, loss_train, loss_test, acc_train, acc_test, epoch):
        if acc_train is None:
            print('epoch: %4d, loss: %2.6f, test loss: %2.6f'                  % (epoch + 1, loss_train, loss_test))
        else:
            print('epoch: %4d, loss: %2.6f, acc: %3.3f,                  test loss: %2.6f, test acc: %3.3f'                  % (epoch + 1, loss_train, acc_train, loss_test, acc_test))

