# Authors: Daichi Yoshikawa <daichi.yoshikawa@gmail.com>
# License: BSD 3 clause

import os, sys
import matplotlib.pyplot as plt
import pickle
import time

import logging
logger = logging.getLogger('dnnet.log')

import dnnet
from dnnet.exception import DNNetIOError, DNNetRuntimeError
from dnnet.ext_mathlibs import cp, np
from dnnet.utils.nn_utils import prod, shuffle_data, split_data, w2im
from dnnet.utils.nn_utils import is_multi_channels_image, flatten, unflatten
from dnnet.training.back_propagation import BackPropagation
from dnnet.layers.layer import Layer, InputLayer, OutputLayer


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
            raise DNNetIOError(msg)

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
        logger.info('Define network with dnnet of version : %s'\
                    % dnnet.__version__)
        if self.layers.size == 0:
            msg = 'NeuralNetwork has no layer.\n Add layers before compiling.'
            raise DNNetRuntimeError(msg)

        parent = self.layers[0]
        self.add(OutputLayer())

        for i, layer in enumerate(self.layers, 1):
            logger.debug('Add %s layer.' % layer.get_type())
            layer.set_parent(parent)
            parent = layer

        logger.debug('Defined network.')

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
        loss_function : Derived class of LossFunction
            Used to calculate loss.
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
        train_data_ratio_for_eval : float, default 1.0
            Ratio of training data to calculate accuracy w.r.t training data.

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
        start = time.time()

        epochs = kwargs.pop('epochs', 10)
        batch_size = kwargs.pop('batch_size', 100)
        learning_curve = kwargs.pop('learning_curve', True)
        shuffle = kwargs.pop('shuffle', True)
        shuffle_per_epoch = kwargs.pop('shuffle_per_epoch', False)
        test_data_ratio = kwargs.pop('test_data_ratio', self.dtype(0.))
        train_data_ratio_for_eval = kwargs.pop(
                'train_data_ratio_for_eval', 1.0)

        logger.info('\n--- Parameters ---\nepochs: %d\nbatch_size: %d\n'
                    'learning_curve: %r\nshuffle: %r\nshuffle_per_epoch: %r\n'
                    'test_data_ratio: %f\ntest_data_ratio_for_eval: %f\n'
                    'optimizer: %s\nloss_function: %s'
                    
                    % (epochs, batch_size, learning_curve, shuffle,
                       shuffle_per_epoch, test_data_ratio,
                       train_data_ratio_for_eval, optimizer.get_type(),
                       loss_function.get_type()))

        if shuffle:
            logger.debug('shuffle data.')
            x, y = shuffle_data(x, y)
        x, y = self.__convert_dtype(x, y)
        x_train, y_train, x_test, y_test = split_data(x, y, test_data_ratio)
        logger.info('Train data input, output : %s, %s'
                    % (x_train.shape, y_train.shape))
        logger.info('Test data input, output : %s, %s'
                    % (x_test.shape, y_test.shape))

        back_prop = BackPropagation(
            epochs, batch_size, optimizer, loss_function,
            learning_curve, self.dtype)

        np_err_config = np.seterr('raise')
        try:
            logger.info('Fitting model starts.')
            lc = back_prop.fit(
                self.layers, x_train, y_train, x_test, y_test,
                shuffle_per_epoch, batch_size, train_data_ratio_for_eval)
        except FloatingPointError as e:
            msg = str(e) + '\nOverflow or underflow occurred. '\
                + 'Retry with smaller learning_rate or '\
                + 'larger weight_decay for Optimizer.'
            raise DNNetRuntimeError(msg)
        except Exception as e:
            raise DNNetRuntimeError(e)
        finally:
            np.seterr(
                divide=np_err_config['divide'],
                over=np_err_config['over'],
                under=np_err_config['under'],
                invalid=np_err_config['invalid']
            )

        end = time.time()
        logger.info('Fitting model is done. '
                    'Processing time : %.2f[s]\n' % (end - start))

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

    def get_config_str(self):
        config_str = ''
        for i, layer in enumerate(self.layers):
            config_str += layer.get_config_str() + '\n'
        config_str = config_str.rstrip('\n')
        return config_str

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
            with open(os.path.join(path, name), 'wb') as f:
                pickle.dump(self, f)
        except IOError as e:
            msg = str(e) + '\nNeuralNetwork.save failed.'
            raise DNNetIOError(msg)

    def visualize_filters(
            self, index, n_rows, n_cols, filter_shape, figsize=(8, 8)):
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
            If this value is out of range, raise DNNetRuntimeError.
        shape : tuple (rows, cols)
            Shape of filter. In the case of multi-channel, filters are
            taken as single channel by taking average over channels.
        filter_shape : tuple (rows, cols)

        layout : tuple (rows, cols)
            Number of filter to display
            in direction of rows and cols respectively.
        """
        # Get index of layer which is index-th layer with weight matrix.
        n_layers_w_filter = 0
        tgt_layer_idx = None
        tgt_layer_type = None

        for i, layer in enumerate(self.layers, 0):
            if layer.has_weight():
                if n_layers_w_filter == index:
                    tgt_layer_idx = i
                    tgt_layer_type = layer.get_type()
                    break
                n_layers_w_filter += 1

        if tgt_layer_idx is None:
            msg = str(index) + '-th layer with weight matrix doesn\'t exist.'
            raise DNNetRuntimeError(msg)
        if tgt_layer_type == 'convolution':
            self.visualize_filter_of_convolution_layer(
                self.layers[tgt_layer_idx], n_rows, n_cols, filter_shape, figsize)
        elif tgt_layer_type == 'affine':
            self.visualize_filter_of_affine_layer(
                self.layers[tgt_layer_idx], n_rows, n_cols, filter_shape, figsize)
        else:
            msg = 'NeuralNetwork.visualize_filters does not support '\
                + '%s' % tgt_layer_type
            raise DNNetRuntimeError(msg)
        print(tgt_layer_idx, tgt_layer_type)

    def visualize_filter_of_convolution_layer(
            self, layer, n_rows, n_cols, filter_shape, figsize=(8, 8)):
        n_filters = layer.w.shape[1]
        if n_filters < n_rows * n_cols:
            msg = 'n_rows and n_cols is too big.\n'\
                + 'n_filters : %d\n' % n_filters\
                + 'n_rows : %d\n' % n_rows\
                + 'n_cols : %d\n' % n_cols
            raise DNNetRuntimeError(msg)

        w = layer.w[1:, :n_rows*n_cols]
        img = w.T.reshape(-1, filter_shape[0], filter_shape[1])
        plt.figure(figsize=figsize)
        plt.imshow(img)
        plt.show()

    def visualize_filter_of_affine_layer(
            self, layer, n_rows, n_cols, filter_shape, figsize=(8, 8)):
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
            If this value is out of range, raise DNNetRuntimeError.
        shape : tuple (rows, cols)
            Shape of filter. In the case of multi-channel, filters are
            taken as single channel by taking average over channels.
        layout : tuple (rows, cols)
            Number of filter to display
            in direction of rows and cols respectively.
        """
        w = layer.w

        if (w.shape[0] - 1) != prod(shape):
            msg = '(w.shape[0] - 1) != prod(shape)\n'\
                + 'w.shape[0] : %d\n' % w.shape[0]\
                + 'prod(shape) : %d' % prod(shape)
            raise DNNetRuntimeError(msg)

        #if w.shape[1] < prod(layout):
        #img = w2im(self.layers[tgt_index].w, shape, layout)
        #plt.figure(figsize=figsize)
        #plt.imshow(img)
        #plt.show()

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
            raise DNNetRuntimeError(msg)

        img = w2im(self.layers[tgt_index].w, shape, layout)
        plt.figure(figsize=figsize)
        plt.imshow(img)
        plt.show()

    def __convert_dtype(self, x, y):
        """Convert data type of features into selected one in constructor."""
        return x.astype(self.dtype), y.astype(self.dtype)
