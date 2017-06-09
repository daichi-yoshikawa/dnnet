
# coding: utf-8

# In[1]:

# Authors: Daichi Yoshikawa <daichi.yoshikawa@gmail.com>
# License: BSD 3 clause

# This file is going to be merged with nn_utils module.

from __future__ import absolute_import

import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.stride_tricks import as_strided


# In[2]:

def pad_img(img, pad_rows, pad_cols):
    """Returns padded matrix which represents image.

    1d matrix is not supported.
    Shape must be in forms of (***, ***, ... , ***, rows, cols),
    such as (rows, cols), (channels, rows, cols),
    or (batch size, channels, rows, cols), etc.

    Arguments
    ---------
    img : np.array
        Image matrix in 2 or more dimensional array.
        This array is not exposed to side effect.
    pad_rows : int or tuple (pad_upper, pad_lower)
        Number of pad in direction of rows.
        If tuple, the first entry is a number of pad in upper side
        and the second one is that in lower side.
    pad_cols : int or tuple (pad_left, pad_right)
        Number of pad in direction of cols.
        If tuple, the first entry is a number of pad in left side
        and the second one is that in right side.

    Returns
    -------
    np.array
        The resulting matrix, that is, padded matrix.
    """
    if (img.ndim < 2):
        msg = '1d array is not supported.'
        raise RuntimeError(msg)

    if np.prod(pad) == 0:
        return img

    npad = ()
    for i in range(img.ndim - 2):
        npad = npad + ((0, 0),)

    if isinstance(pad_rows, tuple):
        npad = npad + ((pad_rows[0], pad_rows[1]),)
    else:
        npad = npad + ((pad_rows, pad_rows),)

    if isinstance(pad_cols, tuple):
        npad = npad + ((pad_cols[0], pad_cols[1]),)
    elif pad_cols:
        npad = npad + ((pad_cols, pad_cols),)

    return np.pad(img, pad_width=npad, mode='constant', constant_values=0)


# In[3]:

def reshape_img(img):
    """Returns reshaped 4d matrix.

    im2col function assumes that input matrix is 4d (bathes, channels, rows, cols).
    This function helps im2col by reshaping the matrix properly.
    If matrix's dimension is more than 4, this throws exception.

    Arguments
    ---------
    img : np.array
        Matrix in 1-4d array.

    Returns
    -------
    np.array
        Matrix in 4d array.
    """
    if len(img.shape) == 1:
        return img.reshape(1, 1, 1, img.shape[0])
    elif len(img.shape) == 2:
        return img.reshape(1, 1, img.shape[0], img.shape[1])
    elif len(img.shape) == 3:
        return img.reshape(1, img.shape[0], img.shape[1], img.shape[2])
    elif len(img.shape) > 4:
        msg = 'len(img.shape) must be <= 4.'
        raise RuntimeError(msg)
    return img


def get_remainders_of_filtering(rows, cols, f_rows, f_cols, strides):
    """Get remainders which resulted from applying filter.

    Combination of image size, filter size and stride size should be proper.
    If it is improper, remainders appear when filtering, that is,
    filter can't be applied to all pixels.
    The resulting remainders can be used to detect applicability of filter,
    or pad image to enable filtering.

    Arguments
    ---------
    img 
    """
    rem_r = (rows - f_rows) % strides[0]
    rem_c = (cols - f_cols) % strides[1]
    return rem_r, rem_c


def extend_img_for_filtering(img, rem_r, rem_c):
    """Extend(Pad) image matrix to enable it to be filtered properly.

    Based on remainders of filtering, pad image with 0s.
    This remainders are supposed to be gained through
    get_remainders_of_filtering function.

    Arguments
    ---------
    img : np.array
        Matrix in 2-4d array, whose shape is (rows, cols), (channels, rows, cols),
        or (batches, channels, rows, cols).
    rem_r : int
        Remainder in rows direction, which is derived from
        get_remainders_of_filtering function.
    rem_c : int
        Remainder in cols direction, which is derived from
        get_remainders_of_filtering function.

    Returns
    -------
    np.array
        Padded image in 2-4d array.
    """
    pad_r = (gap_r//2, gap_r - (gap_r//2))
    pad_c = (gap_c//2, gap_c - (gap_c//2))
    return pad_img(img, pad_r, pad_c)


def im2col(img, f_shape, pad, strides, force=False):
    """Convert 2-4d image matrix into a form which is proper for convolution.

    Convolutional neural network requires convolution and convolution requires
    filtering to 2d images.
    To do it with matrix computation, we have to convert original matrix,
    whose shape would be (rows, cols), (channels, rows, cols)
    or (batches, channels, rows, cols), into different form.

    Arguments
    ---------
    img : np.array
        Image matrix in 2-4d array, whose shape is (rows, cols),
        (channels, rows, cols), or (batches, channels, rows, cols).
    f_shape : tuple (rows, cols)
        Filter's shape.
    pad : tuple (rows, cols)
        Number of pad, which consists of 0s. If rows/cols shape is
        tuple (upper/left, lower/right), you can specify number of pad
        in upper/left or lower/right part of img.
        Eg. 
        img : 1, 2, 3
              4, 5, 6
        In case of pad=(1, 1) :
              0, 0, 0, 0, 0
              0, 1, 2, 3, 0
              0, 4, 5, 6, 0
              0, 0, 0, 0, 0
        In case of pad=((1, 2), (3, 2))
              0, 0, 0, 0, 0, 0, 0, 0
              0, 0, 0, 1, 2, 3, 0, 0
              0, 0, 0, 4, 5, 6, 0, 0
              0, 0, 0, 0, 0, 0, 0, 0
              0, 0, 0, 0, 0, 0, 0, 0
    strides : tuple (rows, cols)
        Stride size of filter in rows and cols direction.
    force : bool, default False
        Force conversion by padding in case of that
        combination of image shape, filter shape and strides is improper.
    """
    pimg = pad_img(reshape_img(img), pad[0], pad[1])

    batches, chs, rows, cols = pimg.shape
    f_rows, f_cols = f_shape
    gap_r, gap_c = get_remainders_of_filtering(rows, cols, f_rows, f_cols, strides)

    if (gap_r > 0) or (gap_c > 0):
        if force:
            pimg = extend_img_for_filtering(pimg, gap_r, gap_c)
            batches, chs, rows, cols = pimg.shape
        else:
            msg = 'Filter cannot be applied to image with the strides.\n'                + 'Image shape (with pad) : ' + str((rows, cols)) + '\n'                + 'Filter shape : ' + str((f_rows, f_cols)) + '\n'                + 'Strides : ' + str(strides)
            raise RuntimeError(msg)

    st_batch, st_ch, st_r, st_c = pimg.strides
    f_st_r = st_r * strides[0]
    f_st_c = st_c * strides[1]
    dst_strides = (st_batch, st_ch, f_st_r, f_st_c, st_r, st_c)

    dst_rows = (rows - f_rows) // strides[0] + 1
    dst_cols = (cols - f_cols) // strides[1] + 1
    dst_shape = (batches, chs, dst_rows, dst_cols, f_rows, f_cols)

    dst_img = as_strided(pimg, shape=dst_shape, strides=dst_strides)
    dst_rows = batches * dst_rows * dst_cols
    dst_cols = chs * f_rows * f_cols
    dst_img = dst_img.transpose(0, 2, 3, 1, 4, 5).reshape(dst_rows, dst_cols)

    return dst_img


# In[5]:

"""
img = np.arange(144).reshape(2, 2, 6, 6).astype(np.float32)

# Arguments
f_shape = (3, 3)
pad = (0, 0)
strides = (1, 1)
force = True

conv_img = im2col(img, f_shape, pad, strides, force)
print(img.shape)
print(img)
print('')
print(conv_img.shape)
print(conv_img)
"""
print()


# In[112]:

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
    batch_size, channels, rows, cols = img.shape
    return img.reshape(batch_size, channels*rows*cols)

