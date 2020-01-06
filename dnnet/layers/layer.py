# Authors: Daichi Yoshikawa <daichi.yoshikawa@gmail.com>
# License: BSD 3 clause

from dnnet.ext_mathlibs import cp, np


class Layer:
    """Base class for layers.

    Parameters
    ----------
    dtype : type
        Data type to use.
    fire : np.array
        Result of forward calculation in this layer.
    backfire : np.array
        Result of backward calculation in this layer.
    parent : Derived class of Layer
        Parent layer of this layer.
    child : Derived class of Layer
        Child layer of this layer.
    input_shape : int or tuple
        Number of neurons of parent's layer.
    output_shape : int or tuple
        Number of neurons of this layer.

    Warning
    -------
    This class should not be used directly.
    Use derived classes instead.
    """
    dtype = np.float64

    def __init__(self):
        self.layer_index = 0
        self.input_shape = None
        self.output_shape = None
        self.fire = None
        self.backfire = None

    def set_dtype(self, dtype):
        pass

    def get_type(self):
        raise NotImplementedError('Layer.get_type')

    def get_config_str(self):
        head = self.get_config_str_head()
        tail = self.get_config_str_tail()
        config_str = head + ', ' * (len(tail.replace(' ', '')) > 0) + tail
        return config_str

    def get_config_str_head(self):
        head = '%d-th %s' % (self.layer_index, self.get_type())
        return head

    def get_config_str_tail(self):
        tail = 'input: %s, ' % (self.input_shape,)
        tail += 'output: %s' % (self.output_shape,)
        return tail

    def has_weight(self):
        return False

    def set_parent(self, parent):
        """Set parent layer to this layer and
        set this layer to parent layer's child layer.

        Arguments
        ---------
        parent : Derived class of Layer
            Any reasonable layer.
        """
        self.parent = parent
        parent.child = self

        self.input_shape = parent.output_shape
        self.layer_index = self.parent.layer_index + 1

    def forward(self, x):
        """Forward calculation called in training phase.

        Arguments
        ---------
        x : np.array
            Fire of parent layer in 2d array.
            If no parent layer, it would be normalized descriptive features.
        """
        raise NotImplementedError('Layer.forward')

    def backward(self, dy):
        """Backward calculation.

        Arguments
        ---------
        dy : np.array
            Backfire of child layer in 2d array.
            If no child layer, it would be errors of
            current predicted results against training data.
        """
        raise NotImplementedError('Layer.backward')

    def predict(self, x):
        """Forward calculation called in prediction phase.

        Arguments
        ---------
        x : np.array
            Fire of parent layer in 2d array.
            If no parent layer, it would be normalized descriptive features.
        """
        raise NotImplementedError('Layer.predict')


class InputLayer(Layer):
    """Implement the first layer of neural network.

    Derived class of Layer.
    """
    def __init__(self, input_shape):
        self.layer_index = 0
        self.input_shape = input_shape
        self.output_shape = self.input_shape

    def get_type(self):
        return 'input'

    def forward(self, x):
        """Starting point of forward calculation."""
        self.child.forward(x)

    def backward(self, dy):
        pass

    def predict(self, x):
        return self.child.predict(x)


class OutputLayer(Layer):
    """Implements the last layer of neural network.

    Derived class of Layer.
    """
    def __init__(self):
        self.layer_index = 0

    def get_type(self):
        return 'output'

    def set_parent(self, parent):
        Layer.set_parent(self, parent)
        self.output_shape = self.input_shape

    def forward(self, x):
        self.fire = x

    def backward(self, dy):
        """Starting point of backward calculation."""
        self.backfire = dy
        self.parent.backward(self.backfire)

    def predict(self, x):
        self.fire = x
        return self.fire
