# Authors: Daichi Yoshikawa <daichi.yoshikawa@gmail.com>
# License: BSD 3 clause

import dnnet.utils.numcupy as ncp
from dnnet.ext_mathlibs import cp, np
from dnnet.exception import DNNetRuntimeError
from dnnet.layers.layer import Layer
from dnnet.training.weight_initialization import DefaultInitialization
from dnnet.utils.nn_utils import asnumpy
from dnnet.utils.cnn_utils import pad_img, im2col, col2im


class ConvolutionLayer(Layer):
    def __init__(
            self, filter_shape, pad=(0, 0), strides=(1, 1),
            weight_initialization=DefaultInitialization()):
        self.filter_shape = filter_shape
        self.pad = pad
        self.strides = strides
        self.weight_initialization = weight_initialization
        self.x = None

    def set_dtype(self, dtype):
        self.dtype = dtype

    def get_type(self):
        return 'convolution'

    def get_config_str_tail(self):
        tail = Layer.get_config_str_tail(self) + ', '
        tail += 'filter: %s, ' % (self.filter_shape,)
        tail += 'pad: %s, ' % (self.pad,)
        tail += 'strides: %s' % (self.strides,)
        return tail

    def set_parent(self, parent):
        Layer.set_parent(self, parent)

        self.__check_shape(self.input_shape)
        self.__init_weight(parent)
        self.__set_output_shape()

    def has_weight(self):
        return True

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
        x = cp.array(x)

        if len(x.shape) != 4:
            msg = 'Convolution layer assumes that input is 4-d array.\n'\
                + '    shape : %s' % str(x.shape)
            raise DNNetRuntimeError(msg)

        n_batches, _, _, _ = x.shape
        n_channels, n_rows, n_cols = self.output_shape

        x_pad = pad_img(x, self.pad[0], self.pad[1])
        x = im2col(x_pad, self.filter_shape, self.strides)
        x = cp.c_[cp.ones((x.shape[0], 1), dtype=self.dtype), x]

        fire = cp.dot(x, cp.array(self.w))
        fire = fire.reshape(n_batches, n_rows, n_cols, n_channels)
        fire = fire.transpose(0, 3, 1, 2)

        self.x = asnumpy(x)
        self.fire = asnumpy(fire)

    def __backward(self, dy):
        dy = cp.array(dy)

        n_batches, _, _, _ = self.fire.shape
        n_channels, n_rows, n_cols = self.input_shape
        n_filters, n_rows_filter, n_cols_filter = self.filter_shape
        dy = dy.transpose(0, 2, 3, 1).reshape(-1, n_filters)

        input_shape = (n_batches, n_channels, n_rows, n_cols)
        backfire = np.dot(dy, cp.array(self.w[1:, :]).T)

        backfire = col2im(
            backfire, input_shape, self.output_shape,
            self.filter_shape, self.pad, self.strides, aggregate=True)

        if self.pad[0] > 0:
            backfire = backfire[:, :, self.pad[0]:-self.pad[0], :]
        if self.pad[1] > 0:
            backfire = backfire[:, :, :, self.pad[1]:-self.pad[1]]

        self.backfire = asnumpy(backfire)
        self.dw = asnumpy(self.dtype(1.) / n_batches * cp.dot(cp.array(self.x).T, dy))

    def __check_shape(self, shape):
        if not isinstance(shape, tuple):
            msg = 'Invalid type of shape : ' + type(shape)
            raise DNNetRuntimeError(msg)
        elif len(shape) != 3:
            msg = 'Invalid shape : ' + str(shape)\
                + '\nShape must be (channels, rows, cols).'
            raise DNNetRuntimeError(msg)

    def __init_weight(self, parent):
        n_channels, _, _ = self.input_shape
        n_filters, n_rows_filter, n_cols_filter = self.filter_shape

        n_rows = n_channels * n_rows_filter * n_cols_filter
        n_cols = n_filters

        self.w = self.weight_initialization.get(n_rows, n_cols, self).astype(self.dtype)
        self.w = np.r_[np.zeros((1, n_cols)), self.w]
        self.w = self.w.astype(self.dtype)
        self.dw = np.zeros_like(self.w, dtype=self.dtype)

    def __set_output_shape(self):
        n_channels_in, n_rows_in, n_cols_in = self.input_shape
        n_channels_filter, n_rows_filter, n_cols_filter = self.filter_shape

        n_channels_out = n_channels_filter
        rem_rows = n_rows_in + 2*self.pad[0] - n_rows_filter
        rem_rows %= self.strides[0]
        rem_cols = n_cols_in + 2*self.pad[1] - n_cols_filter
        rem_cols %= self.strides[1]

        if (rem_rows > 0) or (rem_cols > 0):
            msg = 'Invalid combos of input, filter, pad, and stride.\n'\
                + '    input shape : %s\n' % str(self.input_shape)\
                + '    filter shape : %s\n' % str(self.filter_shape)\
                + '    pad, stride : %s, %s'\
                % (str(self.pad), str(self.strides))
            raise DNNetRuntimeError(msg)

        n_rows_out = n_rows_in + 2*self.pad[0] - n_rows_filter
        n_rows_out = n_rows_out // self.strides[0] + 1
        n_cols_out = n_cols_in + 2*self.pad[1] - n_cols_filter
        n_cols_out = n_cols_out // self.strides[1] + 1

        self.output_shape = (n_channels_out, n_rows_out, n_cols_out)
