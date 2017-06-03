
# coding: utf-8

# In[3]:

# Authors: Daichi Yoshikawa <daichi.yoshikawa@gmail.com>
# License: BSD 3 clause

from __future__ import absolute_import

import numpy as np

from .utils.nn_utils import get_kwarg, shuffle_data, split_data
from .training.random_weight import RandomWeight
from .training.back_propagation import BackPropagation
from .layers.layer import Layer


# In[2]:

class NeuralNetwork:
    """Interface of neural network.

    Training of model and prediction with resulting model
    is done through this class.

    Parameters
    ----------
    layers : np.array of derived class of Layer
        Layers to build neural network.
        The first layer must be InputLayer and last layer must be OutputLayer.
    dtype : type
        Data type selected through constructor.
    """
    def __init__(self, dtype=np.float32):
        """
        Arguments
        ---------
        dtype : type, default np.float32
            Data type to use.
        """
        self.layers = np.array([], dtype=Layer)
        self.dtype = dtype

    def add(self, layer):
        """Add instance of derived class of layer.

        Build neural network by adding layers one by one with this method.

        Arguments
        ---------
        layer : Derived class of Layer
            Instance of derived class of Layer.
        """
        layer.set_dtype(self.dtype)
        self.layers = np.append(self.layers, layer)

    def compile(self):
        """Finalize configuration of neural network model.

        Warning
        -------
        This method must be called after adding all required layers
        and before starting training.
        """
        if self.layers.size == 0:
            msg = 'NeuralNetwork has no layer.\n'                + 'Please add layers before compiling.'
            raise RuntimeError(msg)

        parent = self.layers[0]

        for i, layer in enumerate(self.layers, 1):
            layer.set_parent(parent)
            parent = layer

    def fit(self, x, y, optimizer, loss_function, **kwargs):
        """Train model.

        Arguments
        ---------
        x : np.array
            Descriptive features in 2d array,
            whose shape is (num of data, num of feature)
        y : np.array
            Target features in 2d array,
            whose shape is (num of data, num of feature)
        optimizer : Derived class of Optimizer
            Instance of derived class of Optimizer.
        loss_function : LossFunction.Type
            Type of loss function to use.
        epochs : int, default 10
            Number of iterations of training.
            1 iteration scans all batches one time.
        batch_size : int, default 100
            Dataset is splitted into multiple mini batches
            whose size is this.
        monitor : bool, default True
            Print out evaluation results of ongoing training.
        shuffle : bool, default True
            Shuffle dataset one time before training.
        shuffle_per_epoch : bool, default False
            Shuffle training data every epoch.
        test_data_ratio : float, default 0
            Ratio of test data. If 0, all data is used for training.

        Warning
        -------
        This method assumes that x and y include all data you use.
        If your data set is so large that all data cannot be stored in memory,
        you cannot use this method. Use fit_gen instead.
        """
        epochs = get_kwarg(
                key='epochs',
                dtype=int,
                default_value=10,
                **kwargs)
        batch_size = get_kwarg(
                key='batch_size',
                dtype=int,
                default_value=100,
                **kwargs)
        monitor = get_kwarg(
                key='monitor',
                dtype=bool,
                default_value=True,
                **kwargs)
        shuffle = get_kwarg(
                key='shuffle',
                dtype=bool,
                default_value=True,
                **kwargs)
        shuffle_per_epoch = get_kwarg(
                key='shuffle_per_epoch',
                dtype=bool,
                default_value=False,
                **kwargs)
        test_data_ratio = get_kwarg(
                key='test_data_ratio',
                dtype=self.dtype,
                default_value=self.dtype(0.),
                **kwargs)

        if shuffle:
            x, y = shuffle_data(x, y)
        x, y = self.__convert_dtype(x, y)
        x_train, y_train, x_test, y_test = split_data(x, y, test_data_ratio)

        back_prop = BackPropagation(epochs, batch_size, optimizer, loss_function, monitor, self.dtype)
        back_prop.fit(self.layers, x_train, y_train, x_test, y_test, shuffle_per_epoch)

    def fit_gen(self, x, y, optimizer, loss_function, **kwargs):
        """Train model for large size data set by using generator.
        TODO
        """
        raise NotImplementError('NeuralNetwork.fit_one_batch')

    def predict(self, x):
        """Returns predicted result.

        Arguments
        ---------
        x : np.array
            Discriptive features in 2d array,
            whose shape is (num of data, num of features)

        Returns
        -------
        np.array
            Predicted target features in 2d array,
            whose shape is (num of data, num of features)
        """
        return self.layers[0].predict(x.astype(self.dtype))

    def print_config(self):
        """Display configuration of layers."""
        i = 1
        layer = self.layers[0]

        while layer is not None:
            print(str(i) + '-th layer : ' + layer.get_type())
            layer = layer.child
            i += 1

    def __convert_dtype(self, x, y):
        """Convert data type of features into selected one in constructor."""
        return x.astype(self.dtype), y.astype(self.dtype)


# In[ ]:



