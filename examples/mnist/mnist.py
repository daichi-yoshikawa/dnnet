# Authors: Daichi Yoshikawa <daichi.yoshikawa@gmail.com>
# License: BSD 3 clause

import sys
sys.path.append('../..')

import pickle
import numpy as np
import dnn
from dnn.neuralnet import NeuralNetwork
from dnn.utils.nn_utils import scale_normalization

from dnn.training.optimizer import AdaGrad
from dnn.training.weight_initialization import DefaultInitialization, He
from dnn.training.loss_function import LossFunction

from dnn.layers.affine import AffineLayer
from dnn.layers.activation import Activation, ActivationLayer
from dnn.layers.dropout import DropoutLayer
from dnn.layers.batch_norm import BatchNormLayer

def get_mnist():
    sys.stdout.write('Load MNIST data .')
    x = np.load('input1.npy')
    sys.stdout.write('.')
    x = np.r_[x, np.load('input2.npy')]
    sys.stdout.write('.')
    x = np.r_[x, np.load('input3.npy')]
    sys.stdout.write('.')
    x = np.r_[x, np.load('input4.npy')]
    sys.stdout.write('.')
    x = np.r_[x, np.load('input5.npy')]
    sys.stdout.write('.')
    x = np.r_[x, np.load('input6.npy')]
    sys.stdout.write('.')
    x = np.r_[x, np.load('input7.npy')]
    x = x.astype(float)

    sys.stdout.write('.')
    y = np.load('output.npy')
    y = y.astype(float)

    sys.stdout.write(' Done.\n')

    return x, y

dtype = np.float32
model = NeuralNetwork(input_shape=(1, 28, 28), dtype=dtype)
#model = NeuralNetwork(input_shape=784, dtype=dtype)
model.add(DropoutLayer(drop_ratio=0.2))

model.add(AffineLayer(output_shape=400, weight_initialization=He()))
model.add(BatchNormLayer())
model.add(ActivationLayer(activation=Activation.Type.srrelu))
model.add(DropoutLayer(drop_ratio=0.2))

model.add(AffineLayer(output_shape=400, weight_initialization=He()))
model.add(BatchNormLayer())
model.add(ActivationLayer(activation=Activation.Type.srrelu))

model.add(AffineLayer(output_shape=10, weight_initialization=DefaultInitialization()))
model.add(BatchNormLayer())
model.add(ActivationLayer(activation=Activation.Type.softmax))
model.compile()

model.print_config()

x, y = get_mnist()
scale_normalization(x)
x = x.reshape(-1, 1, 28, 28)

optimizer = AdaGrad(learning_rate=3e-2, weight_decay=1e-3, dtype=dtype)

lc = model.fit(
        x=x,
        y=y,
        epochs=5,
        batch_size=100,
        optimizer=optimizer,
        loss_function=LossFunction.Type.multinomial_cross_entropy,
        learning_curve=True,
        shuffle=True,
        shuffle_per_epoch=True,
        test_data_ratio=0.142857 # Use 60,000 for training and 10,000 for test.
)
lc.plot(figsize=(8,10), fontsize=12)
model.show_filters(0, shape=(28, 28), layout=(10, 10), figsize=(12, 12))

# Auto Encoder
ae = NeuralNetwork(input_shape=(1, 28, 28), dtype=dtype)
ae.add(DropoutLayer(drop_ratio=0.2))

ae.add(AffineLayer(output_shape=100, weight_initialization=He()))
ae.add(BatchNormLayer())
ae.add(ActivationLayer(activation=Activation.Type.srrelu))
ae.add(DropoutLayer(drop_ratio=0.5))

ae.add(AffineLayer(output_shape=784, weight_initialization=He()))
#ae.add(BatchNormLayer())
#ae.add(ActivationLayer(activation=Activation.Type.srrelu))
ae.compile()

ae.print_config()

optimizer = AdaGrad(learning_rate=3e-2, weight_decay=1e-3, dtype=dtype)

x = x.reshape(-1, 1, 28, 28)
y = x.reshape(-1, 784)

lc2 = ae.fit(
        x=x,
        y=y,
        epochs=10,
        batch_size=100,
        optimizer=optimizer,
        loss_function=LossFunction.Type.squared_error,
        learning_curve=True,
        shuffle=True,
        shuffle_per_epoch=True,
        test_data_ratio=0.
)

lc2.plot(figsize=(8, 6), fontsize=12)
ae.show_filters(0, shape=(28, 28), layout=(10, 10), figsize=(12, 12))
