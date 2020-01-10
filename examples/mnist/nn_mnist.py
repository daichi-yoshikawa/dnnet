# Authors: Daichi Yoshikawa <daichi.yoshikawa@gmail.com>
# License: BSD 3 clause

import sys
sys.path.append('../..')

import json
import numpy as np

"""Make sure configure log before importing dnnet."""
import logging.config
with open('../common/logging.json') as f:
    data = json.load(f)
    logging.config.dictConfig(data)

import dnnet
from dnnet.config import Config
Config.enable_gpu()

from dnnet.neuralnet import NeuralNetwork
from dnnet.utils.nn_utils import scale_normalization

from dnnet.training.optimizer import SGD, Momentum, AdaGrad, Adam, AdaDelta, RMSProp, SMORMS3
from dnnet.training.weight_initialization import DefaultInitialization, He
from dnnet.training.loss_function import MultinomialCrossEntropy
from dnnet.training.loss_function import SquaredError

from dnnet.layers.affine import AffineLayer
from dnnet.layers.activation import Activation, ActivationLayer
from dnnet.layers.dropout import DropoutLayer
from dnnet.layers.batch_norm import BatchNormLayer

from data import get_mnist


data_dir = '../../data'
x, y = get_mnist(data_dir)
scale_normalization(x)
x = x.reshape(-1, 1, 28, 28)

dtype = np.float32
force_cpu = {
    'activation': True,
    'dropout': True,
    'batch_norm': True
}

model = NeuralNetwork(input_shape=(1, 28, 28), dtype=dtype)
#model = NeuralNetwork(input_shape=784, dtype=dtype)
model.add(DropoutLayer(drop_ratio=0.2, force_cpu=force_cpu['dropout']))

model.add(AffineLayer(
    output_shape=400, weight_initialization=He()))
model.add(BatchNormLayer(force_cpu=force_cpu['batch_norm']))
model.add(ActivationLayer(activation=Activation.Type.relu,
                          force_cpu=force_cpu['activation']))
model.add(DropoutLayer(drop_ratio=0.2, force_cpu=force_cpu['dropout']))

model.add(AffineLayer(
    output_shape=400, weight_initialization=He()))
model.add(BatchNormLayer(force_cpu=force_cpu['batch_norm']))
model.add(ActivationLayer(activation=Activation.Type.relu,
                          force_cpu=force_cpu['activation']))

model.add(AffineLayer(
    output_shape=10, weight_initialization=DefaultInitialization()))
model.add(BatchNormLayer(force_cpu=force_cpu['batch_norm']))
model.add(ActivationLayer(activation=Activation.Type.softmax,
                          force_cpu=force_cpu['activation']))
model.compile()

config_str = model.get_config_str()
print(config_str)

#optimizer = SGD(learning_rate=3e-2, weight_decay=1e-3, dtype=dtype)
#optimizer = Momentum(learning_rate=3e-2, weight_decay=1e-3, momentum_rate=0.99, dtype=dtype)
optimizer = AdaGrad(learning_rate=3e-2, weight_decay=1e-3, dtype=dtype)
#optimizer = Adam(learning_rate=3e-2, weight_decay=1e-3, dtype=dtype)
#optimizer = AdaDelta(learning_rate=3e-2, weight_decay=1e-3, gamma=0.99, dtype=dtype)
#optimizer = RMSProp(learning_rate=3e-2, weight_decay=1e-3, dtype=dtype)

lc = model.fit(
    x=x, y=y, epochs=5, batch_size=100, optimizer=optimizer,
    loss_function=MultinomialCrossEntropy(),
    learning_curve=True, shuffle=True, shuffle_per_epoch=True,
    test_data_ratio=0.142857, # Use 60,000 for training and 10,000 for test.
    train_data_ratio_for_eval=0.01)
lc.plot(figsize=(8,10), fontsize=12)
model.show_filters(0, shape=(28, 28), layout=(10, 10), figsize=(12, 12))
