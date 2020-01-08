from dnnet.exception import DNNetValueError
from dnnet.ext_mathlibs import cp, np


def random_shuffle(arr):
    if isinstance(arr, np.ndarray):
        np.random.shuffle(arr)
    else:
        cp.random.shuffle(arr)


def mean(arr, axis):
    if isinstance(arr, np.ndarray):
        return np.mean(arr, axis=axis)
    else:
        return cp.mean(arr, axis=axis)


def sqrt(arr):
    if isinstance(arr, np.ndarray):
        return np.sqrt(arr)
    else:
        return cp.sqrt(arr)


def ones(shape, dtype, arr_type):
    if arr_type == type(np.array([])):
        return np.ones(shape, dtype=dtype)
    else:
        return cp.ones(shape, dtype=dtype)


def zeros(shape, dtype, arr_type):
    if arr_type == type(np.array([])):
        return np.zeros(shape, dtype=dtype)
    else:
        return cp.zeros(shape, dtype=dtype)


def power(arr, exponents, dtype=None):
    if isinstance(arr, np.ndarray):
        return np.power(arr, exponents, dtype=dtype)
    else:
        return cp.power(arr, exponents, dtype=dtype)


def exp(arr):
    if isinstance(arr, np.ndarray):
        return np.exp(arr)
    else:
        return cp.exp(arr)


def tanh(arr):
    if isinstance(arr, np.ndarray):
        return np.tanh(arr)
    else:
        return cp.tanh(arr)
    

def pad(array, pad_width, mode='constant', **kwargs):
    if isinstance(array, np.ndarray):
        return np.pad(
            array=array, pad_width=pad_width,
            mode=mode, **kwargs)
    else:
        return cp.pad(array=array, pad_width=pad_width,
                      mode=mode, **kwargs)


def as_strided(x, shape, strides):
    if isinstance(x, np.ndarray):
        return np.lib.stride_tricks.as_strided(
            x=x, shape=shape, strides=strides)
    else:
        return cp.lib.stride_tricks.as_strided(
            x=x, shape=shape, strides=strides)


def concat_by_index_trick(array1, array2, as_new_row=True):
    if type(array1) != type(array2):
        msg = 'ncp.concat_by_index_trick failed. '\
            + 'type(array1) != type(array2)'
        raise DNNetValueError(msg)
    if isinstance(array1, np.ndarray):
        if as_new_row:
            return np.r_[array1, array2]
        else:
            return np.c_[array1, array2]
    else:
        if as_new_row:
            return cp.r_[array1, array2]
        else:
            return cp.c_[array1, array2]


def dot(a, b, out=None):
    if type(a) != type(b):
        msg = 'ncp.dot failed. type(a) != type(b).'
        raise DNNetValueError(msg)
    if isinstance(a, np.ndarray):
        return np.dot(a=a, b=b, out=out)
    else:
        return cp.dot(a=a, b=b, out=out)
