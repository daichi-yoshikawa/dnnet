
# coding: utf-8

# In[1]:

# Authors: Daichi Yoshikawa <daichi.yoshikawa@gmail.com>
# License: BSD 3 clause

from __future__ import absolute_import

import numpy as np
from enum import Enum


# In[4]:

class RandomWeight:
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
        raise NotImplementedError('RandomWeight.get_type')
    
    def get(self, rows, cols):
        """Returns weight set to random values based on selected method.

        Arguments
        ---------
        rows : int
            Number of rows of resulting weight.
        cols : int
            Number of cols of resulting weight.

        Returns
        -------
        np.array
            Resulting weight in 2d array.
        """
        raise NotImplementedError('RandomWeight.get')

    def get_layer_sizes(self, rows, cols):
        """Semantially convert rows and cols into different variables."""
        parent_size = rows
        size = cols
        return parent_size, size


class DefaultRandomWeight(RandomWeight):
    """Default method to randomly initialize weight.

    Reference
    ---------
    Coursera : Machine Learning
    """
    def get_type(self):
        return 'default'

    def get(self, rows, cols):
        parent_size, size = self.get_layer_sizes(rows, cols)
        ep = np.sqrt(6.) / np.sqrt(parent_size + size)
        return ep * np.random.randn(rows, cols)


class Xavier(RandomWeight):
    """Implements Xavier's random initialization.

    Often used with sigmoid, tanh activation functions.

    Reference
    ---------
    Understanding the difficulty of training deep feedforward neural networks
    http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
    """
    def get_type(self):
        return 'xavier'

    def get(self, rows, cols):
        _, size = self.get_layer_sizes(rows, cols)
        ep = 1. / np.sqrt(size)
        return ep * np.random.randn(rows, cols)


class He(RandomWeight):
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

    def get(self, rows, cols):
        _, size = self.get_layer_sizes(rows, cols)
        ep = np.sqrt(2.) / np.sqrt(size)
        return ep * np.random.randn(rows, cols)
    

class RandomWeightFactory:
    """Factory class to get random initialization's instance.

    Warning
    -------
    Get random initialization's instance through this class.
    """
    __random_weight = {
            RandomWeight.Type.default : DefaultRandomWeight(),
            RandomWeight.Type.xavier : Xavier(),
            RandomWeight.Type.he : He(),
    }

    @classmethod
    def get(cls, random_weight):
        """Returns instance of selected random initialization.

        Arguments
        ---------
        random_weight : RandomWeight.Type
            Name of random initialization to use.

        Returns
        -------
        Derived class of RandomWeight
            Instance of selected random initialization.
        """
        return cls.__random_weight[random_weight]


# In[ ]:



