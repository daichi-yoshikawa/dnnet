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
    
