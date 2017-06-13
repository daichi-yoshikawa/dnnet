
# coding: utf-8

# In[ ]:


# Authors: Daichi Yoshikawa <daichi.yoshikawa@gmail.com>
# License: BSD 3 clause

from __future__ import absolute_import

import numpy as np

from .layer import Layer
from ..utils.conv_utils import pad_img, im2col, col2im, im2col_shape
from ..training.random_weight import RandomWeight

class ConvolutionalLayer(Layer):
    def __init__(self, f_shape, pad=(0, 0), strides=(1, 1), force=False):
        self.f_shape = f_shape
        self.pad = pad
        self.strides = strides
        self.force = force
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

    def finalize_training(self, x):
        self.__forward(x)
        self.child.finalize_training(self.fire)

    def __forward(self, x):
        # x.shape : (batches, chs, rows, cols) -> (rows, cols)
        self.x = im2col(x, self.f_shape, self.pad, self.strides, self.force)
        self.fire = np.dot(self.x, self.w) + self.b

        # fire.shape : (rows, cols) -> (batches, chs, rows, cols)
        # Reshape fire into proper shape
        # with consideration of the following description.
        # A column of fire consists of one of resulting images in 1d array.
        # This 1d image is aligned in cols direction, and number of it is
        # the same as the number of filter.
        batches = x.shape[0]
        chs, rows, cols = self.ouput_shape
        self.fire = self.reshape(batches, rows, cols, chs).transpose(0, 3, 1, 2)

    def __backward(self, dy):
        # Transpose and reshape dy. This is the opposite of the
        # reshape and transpose which are applied to fire in __forward method.
        num_of_f, f_rows, f_cols = self.f_shape
        dy = dy.transpose(0, 2, 3, 1).reshape(-1, num_of_f)

        batches, _, _, _ = self.fire.shape
        chs, _, _ = self.input_shape
        self.dw = self.dtype(1.) / bathes * np.dot(self.x.T, dy)
        self.dw = self.dw.T.reshape(num_of_f, chs, f_rows, f_cols)
        self.db = dy.sum(axis=0)

        self.backfire = np.dot(dy, self.w.T)
        ###self.backfire = col2im(self.backfire, FH, FW, self.stride, self.pad)

        #batch_size = self.x.shape[0]
        #self.dw = self.dtype(1.) / batch_size * np.dot(self.x.T, dy)     
        #self.backfire = np.dot(dy, self.w[1:, :].T)

        """
        FN, C, FH, FW = self.W.shape
        dout = dout.transpose(0,2,3,1).reshape(-1, FN)

        self.db = np.sum(dout, axis=0)
        self.dW = np.dot(self.col.T, dout)
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)

        dcol = np.dot(dout, self.col_W.T)
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)
        """
        
    def __check_shape(self, shape):
        if not isinstance(shape, tuple):
            msg = 'Invalid type of shape : ' + type(shape)
            raise RuntimeError(msg)
        elif len(shape) != 3:
            msg = 'Invalid shape : ' + str(shape) + '\nShape must be (channels, rows, cols).'
            raise RuntimeError(msg)

    def __init_weight(self, parent):
        chs, rows, cols = self.input_shape
        num_of_f, f_rows, f_cols = self.f_shape
        w_rows = chs * f_rows * f_cols
        w_cols = num_of_f
        self.w = DefaultRandomWeight().get(w_rows, w_cols)
        self.w = self.w.astype(self.dtype)
        self.dw = np.zeros_like(self.w, dtype=self.dtype)
        self.b = np.zeros((1, w_cols), dtype=self.dtype)
        self.db = np.zeros_like(self.b, dtype=self.dtype)

    def __set_output_shape(self):
        chs = self.f_shape[0]
        rows, cols = im2col_shape(
                self.input_shape, self.f_shape, self.pad, self.strides, self.force)
        self.output_shape = (chs, rows, cols)

