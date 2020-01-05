# Authors: Daichi Yoshikawa <daichi.yoshikawa@gmail.com>
# License: BSD 3 clause

import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.stride_tricks import as_strided

from dnnet.utils.nn_utils import prod


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

    if prod((pad_rows, pad_cols)) == 0:
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


def im2col(img, filter_shape, strides):
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
    f_shape : tuple (num of filter, rows, cols)
        Filter's shape.
    strides : tuple (rows, cols)
        Stride size of filter in rows and cols direction.

    Returns
    -------
    np.array
        The resulting image matrix in 2d array.
    """
    n_batches, n_channels, n_rows, n_cols = img.shape
    _, n_rows_filter, n_cols_filter = filter_shape

    rem_rows = (n_rows - n_rows_filter) % strides[0]
    rem_cols = (n_cols - n_cols_filter) % strides[1]

    if (rem_rows > 0) or (rem_cols > 0):
        msg = 'Filter can not be applied to image '\
            + 'with the specified strides.\n'\
            + '    rem_rows : %d\n    rem_cols : %d' % (rem_rows, rem_cols)
        raise RuntimeError(msg)

    st_batch, st_ch, st_r, st_c = img.strides
    f_st_r = st_r * strides[0]
    f_st_c = st_c * strides[1]
    dst_strides = (st_batch, st_ch, f_st_r, f_st_c, st_r, st_c)

    dst_n_rows = (n_rows - n_rows_filter) // strides[0] + 1
    dst_n_cols = (n_cols - n_cols_filter) // strides[1] + 1
    dst_shape = (n_batches, n_channels, dst_n_rows, dst_n_cols,
                 n_rows_filter, n_cols_filter)
    dst_img = as_strided(img, shape=dst_shape, strides=dst_strides)

    dst_n_rows = n_batches * dst_n_rows * dst_n_cols
    dst_n_cols = n_channels * n_rows_filter * n_cols_filter
    dst_img = dst_img.transpose(
            0, 2, 3, 1, 4, 5).reshape(dst_n_rows, dst_n_cols)

    return dst_img


def col2im(
        col, input_shape, output_shape,
        filter_shape, pad, strides, aggregate=True):
    n_batches, n_channels, n_rows, n_cols = input_shape
    _, n_rows_filter, n_cols_filter = filter_shape
    _, n_rows_out, n_cols_out = output_shape
    n_rows_img = n_rows + 2*prod(pad[0]) + strides[0] - 1
    n_cols_img = n_cols + 2*prod(pad[1]) + strides[1] - 1

    y_strd = strides[0]
    x_strd = strides[1]
    y_step = y_strd * n_rows_out
    x_step = x_strd * n_cols_out

    col_ = col.reshape(
            n_batches, n_rows_out, n_cols_out, n_channels,
            n_rows_filter, n_cols_filter).transpose(0, 3, 4, 5, 1, 2)
    img = np.zeros((n_batches, n_channels, n_rows_img, n_cols_img),
                   dtype=col.dtype)

    if aggregate:
        for y in range(n_rows_filter):
            y_end = y + y_step
            for x in range(n_cols_filter):
                img[:, :, y:y_end:y_strd, x:x+x_step:x_strd] +=\
                        col_[:, :, y, x, :, :]
    else:
        for y in range(n_rows_filter):
            y_end = y + y_step
            for x in range(n_cols_filter):
                img[:, :, y:y_end:y_strd, x:x+x_step:x_strd] =\
                        col_[:, :, y, x, :, :]

    return img
