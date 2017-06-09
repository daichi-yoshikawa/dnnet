
# coding: utf-8

# In[1]:

# Authors: Daichi Yoshikawa <daichi.yoshikawa@gmail.com>
# License: BSD 3 clause

from __future__ import absolute_import

import numpy as np

def get_kwarg(key, dtype, default_value, **kwargs):
    """Get value which is specified through kwargs.

    If the key is not found, default_value is returned.

    Arguments
    ---------
    key : str
        Name of variable.
    dtype : type
        Type of variable.
    default_value : 
        Value which is returned when key is not found.
        Type is the same as the dtype.
    **kwargs : 
        The same as **kwargs.

    Returns
    -------
        If key is found, kwargs[key].
        If key is not found, default_value you set.
    """
    if key in kwargs:
        return dtype(kwargs[key])
    else:
        return default_value


def shuffle_data(x, y):
    """Shuffle descriptive features and target features.

    The 2 matrices must have the same number of rows.
    If not, AttributeError is thrown.

    Arguments
    ---------
    x : np.array
        Descriptive features in 2d array whose shape is (num of data, num of feature).
    y : np.array
        Target features in 2d array whose shape is (num of data, num of feature).

    Returns
    -------
    np.array, np.array
        Rondomly row-shuffled x and y arrays.
    """
    if x.shape[0] != y.shape[0]:
        msg1 = 'x rows : ' + str(x.shape[0]) + '\n'
        msg2 = 'y rows : ' + str(y.shape[0]) + '\n'
        msg = 'x and y data size are different.' * msg1 + msg2
        raise AttributeError(msg)

    index = np.arange(x.shape[0])
    np.random.shuffle(index)

    return x[index], y[index]


def split_data(x, y, test_data_ratio):
    """Split one dataset which consists of descriptive features and target features
    into 2 datasets, that is training data and test data.

    The number of x's row must be the same as the one of y's row.

    Arguments
    ---------
    x : np.array
        Descriptive features in 2d array whose shape is (num of data, num of feature).
    y : np.array
        Target features in 2d array whose shape is (num of data, num of feature).
    test_data_ratio : float
        Desired ratio of test data in range from 0.0 to 1.0. 
        If 0.3, 30% data is for test data and
        rest of the data is for training data.

    Returns
    -------
    np.array, np.array, np.array, np.array
        The former 2 arrays are descriptive features and target features of training data.
        The latter 2 arrays are descriptive features and target features of test data.
    """
    training_data_num = x.shape[0]

    if (test_data_ratio > 0.) and (test_data_ratio < 1.):
        training_data_num = int(training_data_num * (1. - test_data_ratio))

    x_train = x[:training_data_num, :]
    y_train = y[:training_data_num, :]
    x_test = x[training_data_num:, :]
    y_test = y[training_data_num:, :]

    return x_train, y_train, x_test, y_test


def gaussian_normalization(x, ep=1e-5):
    """Normalize 2d matrix to have mean ==0 and standard deviation == 0
    w.r.t each feature.

    This function directly modifies the argument x.

    Arguments
    ---------
    x : np.array
        Features in 2d array whose shape is (num of data, num of feature)
    ep : float
        Used to avoid 0 devision.
    """
    x -= x.mean(axis=0).T
    x /= np.sqrt(x.var(axis=0).T) + ep


def scale_normalization(x, ep=1e-5):
    """Normalize 2d matrix to have values' range from 0.0 to 1.0
    w.r.t each feature.

    This function directly modifies the argument x.

    Arguments
    ---------
    x : np.array
        Features in 2d array whose shape is (num of data, num of feature)
    ep : float
        Used to avoid 0 devision.
    """
    x -= x.min(axis=0).T
    amp = x.max(axis=0) - x.min(axis=0)
    amp = amp.reshape(amp.size, -1)
    x /= (amp.T + ep)


def w2im(w, shape, layout):
    """Reshape 2d weight matrix to 2d image matrix which represents well aligned filters.

    Arguments
    ---------
    w : np.array
        Weight matrix in 2d array.
    shape : tuple (rows, cols)
        Shape of filter. In the case of multi-channel,
        filters are taken as single channel by taking average over channels.
    layout : tuple (rows, cols)
        Number of filter to display in direction of rows and cols respectively.
    """
    if (w.shape[0] - 1) != np.prod(shape):
        msg = '(w.shape[0] - 1) != np.prod(shape)\n'            + '  w.shape[0] : ' + str(w.shape[0]) + '\n'            + '  shape.size : ' + str(np.prod(shape))
        raise RuntimeError(msg)

    if w.shape[1] < np.prod(layout):
        msg = 'w.shape[1] != np.prod(shape)\n'            + '  w.shape[1] : ' + str(w.shape[1]) + '\n'            + '  shape.size : ' + str(np.prod(layout))
        raise RuntimeError(msg)

    img = w[1:, :np.prod(layout)].T
    img = img.reshape(layout[0], layout[1], shape[0], shape[1])
    img = img.transpose(0, 2, 1, 3).reshape(layout[0]*shape[0], layout[1]*shape[1])

    return img


# In[28]:

def is_multi_channels_image(shape):
    if not isinstance(shape, tuple):
        return False

    if len(shape) == 3:
        return True
    return False

def flatten(m):
    batches, chs, rows, cols = m.shape
    return m.reshape(batches, chs*rows*cols)

def unflatten(m, shape_org):
    batches, chs, rows, cols = shape_org
    return m.reshape(batches, chs, rows, cols)


# In[30]:

np.prod(4)


# In[ ]:



