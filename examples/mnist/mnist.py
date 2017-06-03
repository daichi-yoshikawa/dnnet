
# coding: utf-8

# In[19]:

# Authors: Daichi Yoshikawa <daichi.yoshikawa@gmail.com>
# License: BSD 3 clause

from __future__ import absolute_import

import sys
sys.path.append('../../')

import numpy as np
import dnn
from dnn.neuralnet import NeuralNetwork
from dnn.utils.nn_utils import scale_normalization

from dnn.training.optimizer import Adam, AdaGrad, AdaDelta, Momentum
from dnn.training.random_weight import RandomWeight
from dnn.training.loss_function import LossFunction

from dnn.layers.layer import InputLayer, OutputLayer
from dnn.layers.affine import AffineLayer

from dnn.layers.activation import Activation, ActivationLayer
from dnn.layers.dropout import DropoutLayer
from dnn.layers.batch_norm import BatchNormLayer


# In[27]:

def get_mnist():
    x = np.load('input1.npy')
    x = np.r_[x, np.load('input2.npy')]
    x = np.r_[x, np.load('input3.npy')]
    x = np.r_[x, np.load('input4.npy')]
    x = np.r_[x, np.load('input5.npy')]
    x = np.r_[x, np.load('input6.npy')]
    x = np.r_[x, np.load('input7.npy')]
    x = x.astype(float)
    
    y = np.load('output.npy')
    y = y.astype(float)
    
    return x, y


# In[52]:

dtype = np.float32
neuralnet = NeuralNetwork(dtype=dtype)
neuralnet.add(InputLayer(shape=784))
#neuralnet.add(DropoutLayer(drop_ratio=0.2))
neuralnet.add(AffineLayer(shape=(784, 100), random_weight=RandomWeight.Type.he))
neuralnet.add(BatchNormLayer())
neuralnet.add(ActivationLayer(activation=Activation.Type.relu))
#neuralnet.add(DropoutLayer(drop_ratio=0.5))
neuralnet.add(AffineLayer(shape=(100, 10), random_weight=RandomWeight.Type.xavier))
neuralnet.add(BatchNormLayer())
neuralnet.add(ActivationLayer(activation=Activation.Type.softmax))
neuralnet.add(OutputLayer(shape=10))
neuralnet.compile()

x, y = get_mnist()
scale_normalization(x)

optimizer = AdaGrad(learning_rate=3e-2, weight_decay=1e-3, dtype=dtype)

neuralnet.fit(
        x=x,
        y=y,
        epochs=10,
        batch_size=100,
        optimizer=optimizer,
        loss_function=LossFunction.Type.multinomial_cross_entropy,
        monitor=True,
        shuffle=True,
        shuffle_per_epoch=True,
        test_data_ratio=0.142857
)


# In[ ]:



