# Authors: Daichi Yoshikawa <daichi.yoshikawa@gmail.com>
# License: BSD 3 clause

import sys
sys.path.append('../..')

import json
import numpy as np


import logging.config
with open('../common/logging.json') as f:
    data = json.load(f)
    logging.config.dictConfig(data)

from dnnet.config import Config
Config.enable_gpu()

from dnnet.neuralnet import NeuralNetwork

from dnnet.training.optimizer import AdaGrad
from dnnet.training.weight_initialization import DefaultInitialization
from dnnet.training.loss_function import BinomialCrossEntropy

from dnnet.layers.affine import AffineLayer
from dnnet.layers.activation import Activation, ActivationLayer
from dnnet.layers.recurrent_layer import RecurrentLayer
from dnnet.layers.cells.rnn_cell import RNNCell

import logging
logger = logging.getLogger('dnnet.log')


data_dir = '../../data'
#x, y = get_mnist(data_dir)

dtype = np.float32
force_cpu = {
    'activation': False,
}

model = NeuralNetwork(input_shape=(10, 10), dtype=dtype)
model.add(RecurrentLayer(
    cell=RNNCell(), output_shape=100, n_steps=10,
    stack_output=False, stateful=True, force_cpu=False,
    weight_initialization=DefaultInitialization()))
model.add(AffineLayer(
    output_shape=10, weight_initialization=DefaultInitialization()))
model.add(ActivationLayer(activation=Activation.Type.softmax,
                          force_cpu=force_cpu['activation']))
model.compile()

config_str = model.get_config_str()
logger.info(config_str)

optimizer = AdaGrad(learning_rate=1e-3, weight_decay=1e-4, dtype=dtype)

