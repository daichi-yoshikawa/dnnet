# Authors: Daichi Yoshikawa <daichi.yoshikawa@gmail.com>
# License: BSD 3 clause

import numpy as np

from dnn.layers.layer import Layer
from dnn.utils.conv_utils import im2col, col2im


class PoolingLayer(Layer):
    def __init__(self, window_shape):
        self.layer_index = 0
        self.window_shape = window_shape
        self.pad = (0, 0)
        self.strides = (1, 1)

    def set_dtype(self, dtype):
        self.dtype = dtype

    def get_type(self):
        return 'pooling'

    def get_config_str_tail(self):
        tail = Layer.get_config_str_tail(self) + ', '
        tail += 'filter: %s, ' % (self.window_shape,)
        tail += 'pad: %s, ' % (self.pad,)
        tail += 'strides: %s' % (self.strides,)
        return tail

    def set_parent(self, parent):
        Layer.set_parent(self, parent)

        self.__check_shape(self.input_shape)
        self.__set_output_shape()

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
        if len(x.shape) != 4:
            msg = 'Pooling layer only supports 4-d array.\n'\
                + '    input shape : %s' % str(x.shape)
            raise RuntimeError(msg)

        n_batches, _, _, _ = x.shape
        n_channels, n_rows, n_cols = self.output_shape
        window_shape = tuple([1] + list(self.window_shape))

        self.x = im2col(x, window_shape, self.strides)
        self.x = self.x.reshape(self.x.shape[0] * n_channels, -1)
        self.fire = np.max(self.x, axis=1)
        self.fire = self.fire.reshape(
                n_batches, n_rows, n_cols, n_channels).transpose(0, 3, 1, 2)

    def __backward(self, dy):
        indices = [np.arange(self.x.shape[0], dtype=int),
                   np.argmax(self.x, axis=1)]
        grad = np.zeros_like(self.x, dtype=self.x.dtype)
        grad[indices] = 1
        self.backfire = grad * dy.transpose(
                0, 2, 3, 1).flatten().reshape(-1, 1)

        n_batches, _, _, _ = self.fire.shape
        input_shape = tuple([n_batches] + list(self.input_shape))
        window_shape = tuple([1] + list(self.window_shape))

        self.backfire = col2im(
                self.backfire, input_shape, self.output_shape,
                window_shape, self.pad, self.strides, aggregate=True)

    def __check_shape(self, shape):
        if not isinstance(shape, tuple):
            msg = 'Invalid type of shape : ' + type(shape)
            raise RuntimeError(msg)
        elif len(shape) != 3:
            msg = 'Invalid shape : ' + str(shape)\
                + '\nShape must be (channels, rows, cols).'
            raise RuntimeError(msg)

    def __set_output_shape(self):
        n_channels, n_rows_in, n_cols_in = self.input_shape
        n_rows_window, n_cols_window = self.window_shape

        rem_rows = n_rows_in + 2*self.pad[0] - n_rows_window
        rem_rows %= self.strides[0]
        rem_cols = n_cols_in + 2*self.pad[1] - n_cols_window
        rem_cols %= self.strides[1]

        if(rem_rows > 0) or (rem_cols > 0):
            msg = 'Invalid combos of input, window, pad, and stride.\n'\
                + '    input shape : %s\n' % str(self.input_shape)\
                + '    window size : %s\n' % str(self.window_shape)\
                + '    pad, stride : %s, %s'\
                % (str(self.pad), str(self.strides))
            raise RuntimeError(msg)

        n_rows_out = n_rows_in + 2*self.pad[0] - n_rows_window
        n_rows_out = n_rows_out // self.strides[0] + 1
        n_cols_out = n_cols_in + 2*self.pad[1] - n_cols_window
        n_cols_out = n_cols_out // self.strides[1] + 1

        self.output_shape = (n_channels, n_rows_out, n_cols_out)
