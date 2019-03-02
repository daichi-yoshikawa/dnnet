import os
import sys
import numpy as np


def get_mnist(path):
    sys.stdout.write('Load MNIST data ')

    x = np.load(os.path.join(path, 'input1.npy'))
    x = np.r_[x, np.load(os.path.join(path, 'input2.npy'))]
    sys.stdout.write('.')
    x = np.r_[x, np.load(os.path.join(path, 'input3.npy'))]
    sys.stdout.write('.')
    x = np.r_[x, np.load(os.path.join(path, 'input4.npy'))]
    sys.stdout.write('.')
    x = np.r_[x, np.load(os.path.join(path, 'input5.npy'))]
    sys.stdout.write('.')
    x = np.r_[x, np.load(os.path.join(path, 'input6.npy'))]
    sys.stdout.write('.')
    x = np.r_[x, np.load(os.path.join(path, 'input7.npy'))]
    sys.stdout.write('.')
    y = np.load(os.path.join(path, 'output.npy'))
    sys.stdout.write('.')

    sys.stdout.write(' Done.\n')
    return x.astype(float), y.astype(float)
