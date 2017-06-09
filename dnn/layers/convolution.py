
# coding: utf-8

# In[9]:

# Authors: Daichi Yoshikawa <daichi.yoshikawa@gmail.com>
# License: BSD 3 clause

from __future__ import absolute_import

import numpy as np

from .layer import Layer
from ..utils.conv_utils import im2col
from ..training.random_weight import RandomWeight


# In[4]:

class ConvolutionalLayer(Layer):
    def __init__(self, shape, f_shape, pad=(0, 0), strides=(1, 1), force=False):
        self.shape = shape
        self.f_shape = f_shape
        self.pad = pad
        self.strides = strides
        self.force = force


        #self.w = DefaultRandomWeight().get(self.parent_shape, self.shape)
        #self.w = np.r_[np.zeros((1, self.shape)), self.w]
        #self.dw = np.zeros_like(self.w, dtype=self.w.dtype)
        #self.x = None
        
    def get_type(self):
        return 'convolution'

    def set_parent(self, parent):
        Layer.set_parent(self, parent)
        self.shape = parent.shape
        self.__init_weight(parent)
    
    def forward(self, x):
        pass
    
    def backward(self, dy):
        pass
    
    def predict_to_eval(self, x):
        pass
    
    def predict(self, x):
        pass
    
    def finalize_training(self, x):
        pass
    
    def __init_weight(self, parent):
        pass


# In[ ]:

"""
def pad_image(image, pad=(0, 0)):
    '''
    Pad zero values to image.
    
    Attributes
    ----------
    image : np.array
        image.shape is supposed to be
        (rows, cols),
        (channels, rows, cols),
        (batch_size, channels, rows, cols)
    
    pad : Tuple
        Consists of 2 integers.
        The number of pad you'd like to add to row axis and column axis.
    
    Returns
    -------
    np.array
        Padded image.
    '''
    shape_len = len(image.shape)
    
    if shape_len == 2:
        npad = ((pad[0], pad[0]), (pad[1], pad[1]))
    elif shape_len == 3:
        npad = ((0, 0), (pad[0], pad[0]), (pad[1], pad[1]))
    elif shape_len == 4:
        npad = ((0, 0), (0, 0), (pad[0], pad[0]), (pad[1], pad[1]))
    elif shape_len == 5:
        npad = ((0, 0), (0, 0), (0, 0), (pad[0], pad[0]), (pad[1], pad[1]))
    else:
        raise RuntimeError('len(image.shape) > 5 or < 2.')
    return np.pad(image, pad_width=npad, mode='constant', constant_values=0)


def im2col(image, window_shape, pad=(0, 0), strides=(1, 1), ch_axis=0):
    def assert_strides(img_shape, win_shape, pad, strides):
        if (img_shape[0] - win_shape[0]) % strides[0] > 0:
            msg = 'Improper combination of image size, window size '\
                + 'and strides.\n'\
                + '(img_shape[0] - win_shape[0]) % strides[0] '\
                + 'must be 0.'
            raise RuntimeError(msg)
        if (img_shape[1] - win_shape[1]) % strides[1] > 0:
            msg = 'Improper combination of image size, window size '\
                + 'and strides.\n'\
                + '(img_shape[1] - win_shape[1]) % strides[1] '\
                + 'must be 0.'
            raise RuntimeError(msg)
    '''
    image.shape must be (batch_size, channels, rows, cos)
    window_shape must be (rows, cols)
    strides must be (stride_rows, stride_cols)
    '''
    assert_strides(image.shape, window_shape, pad, strides)    

    batch_sizes, channels, img_rows, img_cols = image.shape
    stride_per_batch, stride_per_channel, stride_per_row, stride_per_col\
            = image.strides
    win_rows, win_cols = window_shape
    
    dst_rows = (img_rows - win_rows) // strides[0] + 1
    dst_cols = (img_cols - win_cols) // strides[1] + 1

    # Byte size required to stride running window in row/col direction
    win_stride_per_row = stride_per_row * strides[0]
    win_stride_per_col = stride_per_col * strides[1]

    '''
    ---- How to decide shape ----
    Start from small part and consider bigger part later.
    
    Step 1. Put win_rows and win_cols as arguments
        Resulting image should have (win_rows, win_cols) size matrices
        as the smallest element. This small matrices are reshaped to
        (1, win_rows * win_cols) matrices later.
    Step 2. Put dst_rows and dst_cols as arguments
        The number of the small matrices, we considered at step 1,
        is dst_rows * dst_cols.
    Step 3. Put channels
        One image for one channel has the above matrix, whose size
        is dst_rows * dst_cols and each row have (win_rows, win_cols)
        matrices. Now we consider N-channels.
    Step 4. Put batch_sizes
        Step 3 is for a single image. We use multiple images
        for batch process.
        
    ---- How to decide strides ----
    All you need to do is put stride size which correspond with shape,
    that is, batch_size, channels, dst_rows, dst_cols, win_rows, win_cols.
    '''
    shape = (batch_sizes, channels, dst_rows, dst_cols, win_rows, win_cols)    
    strides = (
            stride_per_batch, stride_per_channel,
            win_stride_per_row, win_stride_per_col,
            stride_per_row, stride_per_col)
    
    dst = as_strided(image, shape=shape, strides=strides)
    '''
    Firstly we have to switch dimensions to reshape dst matrix properly.
    Here are meanings of indices.
    0: Batch size, 
    1: channels,
    2: dst_rows,
    3: dst_cols,
    4: win_rows,
    5: win_cols
    
    We align N-channels data in the same row.
    So we need to move "channels dimension" right after dst_rows and dst_cols.
    By this, we can put (win_rows, win_cols)-sized matrices for N-channels
    to the same dimension later.
    '''
    
    if ch_axis == 0:
        dst_col = win_rows * win_cols
        return dst.reshape(batch_size, channels, -1, dst_col)
    else:
        dst_col = win_rows * win_cols * channels
        return dst.transpose(0, 2, 3, 1, 4, 5).reshape(batch_size, -1, dst_col)


def col2im(mat, window_shape, batch_size, channels, h, w, strides=(1, 1), ch_axis=1):
    win_h, win_w = window_shape

    row_indices = np.arange(0, oh*ow).reshape(oh, -1)[::win_h, ::win_w]
    row_indices = row_indices.reshape(1, -1)[0, :]
    image = mat[:, row_indices, :].reshape(batch_size, 4, channels, win_h*win_w)
    image = image.transpose(0, 2, 1, 3)
    image = image.reshape(batch_size, channels, h//win_h, w//win_w, win_h, win_w)
    image = image.transpose(0, 1, 2, 4, 3, 5).reshape(batch_size, channels, h, w)

    return image


def flatten(img):
    '''
    Expand 4-dimensional image matrix to 2-dimensional matrix.
    The resulting matrix can be used for fully connected layer
    of neural network.
    
    Attributes
    ----------
    img : np.array
        Shape is supposed to be (batch size, channels, rows, cols).
        This array is not modified in this function.
        
    Returns
    -------
    np.array
        2-dimensional matrix, whose shape is (batch size, channels*rows*cols).
        This array is newly created in this function.
    '''
    batch_size, channels, rows, cols = img.shape
    return img.reshape(batch_size, channels*rows*cols)
"""

