# Authors: Daichi Yoshikawa <daichi.yoshikawa@gmail.com>
# License: BSD 3 clause

import logging
logger = logging.getLogger('dnnet.log')

from dnnet.exception import DNNetRuntimeError
from dnnet.ext_mathlibs import cp, np
from dnnet.layers.layer import Layer
from dnnet.training.weight_initialization import DefaultInitialization
from dnnet.utils.nn_utils import prod, asnumpy


class RecurrentLayer(Layer):
    def __init__(
            self, cell, output_shape, n_steps, stack_output=False,
            stateful=False, force_cpu=False,
            weight_initialization=DefaultInitialization()):
        self.cell = cell
        self.output_shape = output_shape
        self.n_steps = n_steps
        self.stack_output = stack_output
        self.stateful = stateful
        self.force_cpu = force_cpu
        self.weight_initialization = weight_initialization

    def set_dtype(self, dtype):
        self.dtype = dtype

    def get_type(self):
        return 'recurrent'

    def set_parent(self, parent):
        """
           Eg1. single channel and N dimensional input features.
           input_shape = (n_steps, n_features)
           eg. n_steps == 3, n_features == 2
           x = [[[x01, x02], [x04, x05], [x07, x08]],
                [[x10, x11], [x12, x13], [x14, x15]],
                ...
               ]

           Eg2. multi channels and N dimensional input features.
           input_shape = (n_steps, n_channels, n_features)
           eg. n_steps == 3, n_channels == 3, n_features = 2
           x = [[X01], [X02], [X03],
                [X04], [X05], [X06],
                ...
               ]
           where X01 = [[x11, x12], [x21, x22], [x31, x32]],
           that is, 3 channels and 2 features for each.

           Eg3. multi channels and MxN dimesnional input features.
           input_shape = (n_steps, n_channels, n_rows, n_cols)
           eg. n_steps = 3, n_channels == 3, n_rows == 2, n_cols == 2
           x = [[X01], [X02], [X03],
                [X04], [X05], [X06],
                ...
               ]
           where X01 = [img11, img12, img13]
                     = [[[x111, x112], [x113, x114]], # <- img11
                        [[x121, x122], [x123, x124]], # <- img12
                        [[x131, x132], [x133, x134]]] # <- img13,
           that is, 3 channels and 2D matrix for each channel.
        """
        Layer.set_parent(self, parent)

        if len(self.input_shape) != 2:
            msg = 'RecurrentLayer must has len(self.input_shape) == 2.\n'\
                + 'self.input_shape : %s' % str(self.input_shape)
            raise DNNetRuntimeError(msg)

        w_rows = prod(self.input_shape[0])
        w_cols = prod(self.output_shape)

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
        self.fire = self.cell.forward(x, self.n_steps)

    def __backward(self, dy):
        self.backfire = self.cell.backward(dy, self.n_steps)
        
