
# coding: utf-8

# In[ ]:


# Authors: Daichi Yoshikawa <daichi.yoshikawa@gmail.com>
# License: BSD 3 clause

from __future__ import absolute_import

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

from .utils.nn_utils import get_kwarg, shuffle_data, split_data, w2im
from .utils.nn_utils import is_multi_channels_image, flatten, unflatten
from .training.random_weight import RandomWeight
from .training.back_propagation import BackPropagation
from .layers.layer import Layer, InputLayer, OutputLayer


# In[ ]:


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
    @classmethod
    def load(self, name, path=None):
        """Load model from storage.

        Arguments
        ---------
        name : str or None, default None
            Name of the desired file. Doesn't include path.
        path : str or None, default None
            Full path to the directory where the desired file is contained.
            If None, file is loaded from a directory where script runs.

        Returns
        -------
        NeuralNetwork
            Returns model.
        """
        if path is None:
            path = '.'
        if path[0] == '~':
            path = os.getenv("HOME") + path[1:]

        try:
            with open(path + '/' + name, 'rb') as f:
                return pickle.load(f)
        except IOError as e:
            msg = str(e) + '\nNeuralNetwork.load failed.'
            print(msg)

    def __init__(self, input_shape, dtype=np.float32):
        """
        Arguments
        ---------
        dtype : type, default np.float32
            Data type to use.
        """
        self.layers = np.array([], dtype=Layer)
        self.dtype = dtype

        self.add(InputLayer(input_shape=input_shape))

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
            msg = 'NeuralNetwork has no layer.\n Add layers before compiling.'
            raise RuntimeError(msg)

        parent = self.layers[0]

        for i, layer in enumerate(self.layers, 1):
            layer.set_parent(parent)
            parent = layer

        output_layer = OutputLayer()
        output_layer.set_parent(self.layers[-1])
        self.add(output_layer)

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
        learning_curve : bool, default True
            Prints out evaluation results of ongoing training.
            Also, returns learning curve after completion of training.
        shuffle : bool, default True
            Shuffle dataset one time before training.
        shuffle_per_epoch : bool, default False
            Shuffle training data every epoch.
        test_data_ratio : float, default 0
            Ratio of test data. If 0, all data is used for training.

        Returns
        -------
        LearningCurve
            Instance of LearningCurve, which contains
            losses and accuracies for train and test data.

        Warning
        -------
        This method assumes that x and y include all data you use.
        If your data set is so large that all data cannot be stored in memory,
        you cannot use this method. Use fit_genenerator instead.
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
        learning_curve = get_kwarg(
                key='learning_curve',
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

        back_prop = BackPropagation(
                epochs,
                batch_size,
                optimizer,
                loss_function,
                learning_curve,
                self.dtype
        )

        lc = back_prop.fit(
                self.layers,
                x_train,
                y_train,
                x_test,
                y_test,
                shuffle_per_epoch
        )

        return lc

    def fit_generator(self, x, y, optimizer, loss_function, **kwargs):
        """Train model for large size data set by using generator.
        TODO(
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
        for i, layer in enumerate(self.layers):
            print(('%2d-th layer, (input_shape, output_shape) : '
                   '({input_shape}, {output_shape}), {layer_type}'
                   .format(input_shape=layer.input_shape,
                           output_shape=layer.output_shape,
                           layer_type=layer.get_type()) % (i)))

    def save(self, name, path=None):
        """Save model to storage.

        Arguments
        ---------
        name : str or None, default None
            Name of the resulting file. Doesn't include path.
        path : str or None, default None
            Full path to the directory where the resulting file is generated.
            If None, file is saved in a directory where script runs.

        Returns
        -------
        bool
            Returns true when succeeded.
        """
        if path is None:
            path = '.'
        if path[0] == '~':
            path = os.getenv("HOME") + path[1:]

        try:
            with open(path + '/' + name, 'wb') as f:
                pickle.dump(self, f)
        except IOError as e:
            msg = str(e) + '\nNeuralNetwork.save failed.'
            print(msg)

    def show_filters(self, index, shape, layout, figsize=(8, 8)):
        """Visualize filters.

        Weight matrix in affine layer or convolution layer
        can be shown as image.
        If weight matrix is so big that all filters cannot be displayed,
        displayed filters are randomly selected.

        Arguments
        ---------
        index : int
            index-th affine/convolution layer's weight matrix is visualized.
            This index starts from 0, that is,
            the first layer with weight matrix is 0-th.
            If this value is out of range, raise RuntimeError.
        shape : tuple (rows, cols)
            Shape of filter. In the case of multi-channel, filters are
            taken as single channel by taking average over channels.
        layout : tuple (rows, cols)
            Number of filter to display
            in direction of rows and cols respectively.
        """
        # Get index of layer which is index-th layer with weight matrix.
        num_of_layer_with_filter = 0
        tgt_index = None

        for i, layer in enumerate(self.layers, 0):
            if layer.has_weight():
                if num_of_layer_with_filter == index:
                    tgt_index = i
                    break
                num_of_layer_with_filter += 1

        if tgt_index is None:
            msg = str(index) + '-th layer with weight matrix doesn\'t exist.'
            raise RuntimeError(msg)

        img = w2im(self.layers[tgt_index].w, shape, layout)
        plt.figure(figsize=figsize)
        plt.imshow(img)
        plt.show()

    def __convert_dtype(self, x, y):
        """Convert data type of features into selected one in constructor."""
        return x.astype(self.dtype), y.astype(self.dtype)

