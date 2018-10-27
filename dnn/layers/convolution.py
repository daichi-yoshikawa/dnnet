# Authors: Daichi Yoshikawa <daichi.yoshikawa@gmail.com>
# License: BSD 3 clause

import numpy as np

from dnn.layers.layer import Layer
from dnn.training.random_weight import RandomWeight, RandomWeightFactory
from dnn.utils.conv_utils import pad_img, im2col, col2im


class ConvolutionalLayer(Layer):
    def __init__(self, filter_shape, pad=(0, 0), strides=(1, 1)):
        self.filter_shape = filter_shape
        self.pad = pad
        self.strides = strides
        self.random_weight = RandomWeightFactory().get(RandomWeight.Type.default)
        self.x = None

    def get_type(self):
        return 'convolution'

    def set_parent(self, parent):
        Layer.set_parent(self, parent)

        self.__check_shape(self.input_shape)
        self.__init_weight(parent)
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
            msg = 'Convolution layer assumes that input is 4-d array.\n'\
                + '    shape : %s' % str(x.shape)
            raise RuntimeError(msg)

        n_batches, _, _, _ = x.shape
        n_channels, n_rows, n_cols = self.output_shape

        x_pad = pad_img(x, self.pad[0], self.pad[1])
        self.x = im2col(x_pad, self.filter_shape, self.strides)
        self.x = np.c_[np.ones((self.x.shape[0], 1), dtype=self.dtype), self.x]

        self.fire = np.dot(self.x, self.w)
        self.fire = self.fire.reshape(n_batches, n_rows, n_cols, n_channels)
        self.fire = self.fire.transpose(0, 3, 1, 2)

    def __backward(self, dy):
        n_batches, _, _, _ = self.fire.shape
        n_channels, _, _ = self.input_shape
        n_filters, n_rows_filter, n_cols_filter = self.filter_shape
        dy = dy.transpose(0, 2, 3, 1).reshape(-1, n_filters)
        self.dw = self.dtype(1.) / n_batches * np.dot(self.x.T, dy)

        input_shape = list(self.fire.shape)
        input_shape[3] -= 1
        output_shape = list(self.output_shape)
        output_shape[1] -= 1
        self.backfire = np.dot(dy, self.w[1:, :].T)

        print(self.backfire.shape, input_shape, output_shape)
        self.backfire = col2im(
                self.backfire, tuple(input_shape), tuple(output_shape),
                self.filter_shape, self.pad, self.strides, aggregate=True)
        self.backfire = self.backfire[:, :, pad[0]:-pad[0], pad[1]:-pad[1]]
        print(self.backfire)

    def __check_shape(self, shape):
        if not isinstance(shape, tuple):
            msg = 'Invalid type of shape : ' + type(shape)
            raise RuntimeError(msg)
        elif len(shape) != 3:
            msg = 'Invalid shape : ' + str(shape) + '\nShape must be (channels, rows, cols).'
            raise RuntimeError(msg)

    def __init_weight(self, parent):
        n_channels, _, _ = self.input_shape
        n_filters, n_rows_filter, n_cols_filter = self.filter_shape

        n_rows = n_channels * n_rows_filter * n_cols_filter + 1
        n_cols = n_filters

        self.w = self.random_weight.get(n_rows, n_cols).astype(self.dtype)
        self.dw = np.zeros_like(self.w, dtype=self.dtype)

    def __set_output_shape(self):
        n_channels_in, n_rows_in, n_cols_in = self.input_shape
        n_channels_filter, n_rows_filter, n_cols_filter = self.filter_shape

        n_channels_out = n_channels_filter
        rem_rows = (n_rows_in + 2*self.pad[0] - n_rows_filter) % self.strides[0]
        rem_cols = (n_cols_in + 2*self.pad[1] - n_cols_filter) % self.strides[1]

        if (rem_rows > 0) or (rem_cols > 0):
            msg = 'Invalid combos of input shape, filter size, pad, and stride.\n'\
                + '    input shape : %s\n' % str(self.input_shape)\
                + '    filter size : %s\n' % str(self.filter_shape)\
                + '    pad, stride : %s, %s' % (str(self.pad), str(self.strides))
            RuntimeError(msg)

        n_rows_out = (n_rows_in + 2*self.pad[0] - n_rows_filter) // self.strides[0] + 1
        n_cols_out = (n_cols_in + 2*self.pad[1] - n_cols_filter) // self.strides[1] + 1

        self.output_shape = (n_channels_out, n_rows_out, n_cols_out)
