# Authors: Daichi Yoshikawa <daichi.yoshikawa@gmail.com>
# License: BSD 3 clause

from dnnet.layers.layer import Layer
from dnnet.ext_mathlibs import cp, np
from dnnet.training.weight_initialization import DefaultInitialization
from dnnet.utils.nn_utils import is_multi_channels_image
from dnnet.utils.nn_utils import prod, ascupy, asnumpy, flatten, unflatten

print(cp)


class AffineLayer(Layer):
    """Implement affine transform of matrix.

    Derived class of Layer.

    Parameters
    ----------
    w : np.array
        Weight in 2d array.
    dw : np.array
        Gradient of weight in 2d array.
    x : np.array
        Extended parent layer's output in 2d array.
        This consists of original parent layer's output and bias term.
    """
    def __init__(self, output_shape, weight_initialization=DefaultInitialization()):
        self.output_shape = output_shape
        self.weight_initialization = weight_initialization
        self.x = None
        self.multi_channels_image = False

    def set_dtype(self, dtype):
        self.dtype = dtype

    def get_type(self):
        return 'affine'

    def set_parent(self, parent):
        Layer.set_parent(self, parent)

        w_rows = prod(self.input_shape)
        w_cols = prod(self.output_shape)

        self.w = self.weight_initialization.get(w_rows, w_cols, self)
        self.w = cp.r_[cp.zeros((1, w_cols)), self.w]
        self.w = self.w.astype(self.dtype)
        self.dw = cp.zeros_like(self.w, dtype=self.w.dtype)

    def has_weight(self):
        return True

    def forward(self, x):
        #self.__forward(x)
        #self.child.forward(self.fire)
        self.__forward(cp.array(x))
        self.child.forward(asnumpy(self.fire))

    def backward(self, dy):
        #self.__backward(dy)
        #self.parent.backward(self.backfire)
        self.__backward(cp.array(dy))
        self.parent.backward(asnumpy(self.backfire))

    def predict(self, x):
        #self.__forward(x)
        #return self.child.predict(self.fire)
        self.__forward(ascupy(x))
        return self.child.predict(asnumpy(self.fire))

    def __forward(self, x):
        if is_multi_channels_image(self.input_shape):
            x = flatten(x, self.input_shape)

        # Add bias terms.
        self.x = cp.c_[cp.ones((x.shape[0], 1), dtype=self.dtype), x]
        self.fire = cp.dot(self.x, self.w)

        if is_multi_channels_image(self.output_shape):
            self.fire = unflatten(self.fire, self.output_shape)

    def __backward(self, dy):
        if is_multi_channels_image(self.output_shape):
            dy = flatten(dy, self.output_shape)

        batch_size = self.x.shape[0]
        self.dw = asnumpy(self.dtype(1.) / batch_size * cp.dot(self.x.T, dy))
        self.backfire = cp.dot(dy, self.w[1:, :].T)

        if is_multi_channels_image(self.input_shape):
            self.backfire = unflatten(self.backfire, self.input_shape)
