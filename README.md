dnnet
=====
Implementation of Deep Neural Network with numpy.
**Now dnnet can run with GPU through cupy.**

dnnet provides high-level API to define and run neural network model.
User can turn on/off GPU layer-wise, that is, you can compute convolution layer with GPU, activation layer with CPU, and dropout layer with CPU, for example.

# Table of Contents
* Brief tour of dnnet; Introduce small examples, supported methodologies
* Installation
* Example; Run sample scripts
* Use in your project


# Brief tour of dnnet
## Quick glance of usage

User can create a instance of NeuralNetwork, add layers one by one,<br>
finalize model, set optimizer, execute model fitting, and save model.

In the below, some arguments are not specified to simplify the example.
```
from dnnet.neuralnet import NeuralNetwork
from dnnet.training.optimizer import AdaGrad
from dnnet.training.weight_initialization import DefaultInitialization, He
from dnnet.training.loss_function import MultinomialCrossEntropy
from dnnet.layers.activation import Activation, ActivationLayer
from dnnet.layers.affine import AffineLayer
from dnnet.layers.batch_norm import BatchNormLayer
from dnnet.layers.convolution import ConvolutionLayer
from dnnet.layers.dropout import DropoutLayer


# Load x, y here

model = NeuralNetwork(input_shape=(1, 28, 28), dtype=np.float32)

model.add(ConvolutionLayer(filter_shape=(32, 3, 3))
model.add(BatchNormLayer())
model.add(ActivationLayer(activation=Activation.Type.relu))
model.add(DropoutLayer(drop_ratio=0.25))

model.add(AffineLayer(output_shape=10)
model.add(ActivationLayer(activation=Activation.Type.softmax)
model.compile()

optimizer = AdaGrad(learning_rate=1e-3, weight_decay=1e-3)
learning_curve = model.fit(
    x=x, y=y, epochs=5, batch=size=100, optimizer=optimizer,
    loss_function=MultinomialCrossEntropy())
model.save(path='./data/output', name='my_cnn.dat')
```

User can also load model, and predict output.
```
model.load(path='./data/output', name='my_cnn.dat')
y_pred = model.predict(x_unknown)
```

GPU is easily enabled. Do the follows at the top of your script.
```
from dnnet.config import Config
Config.enable_gpu()
```

If GPU is enabled but you'd like to turn it off for some specific layers, you can use force_cpu flag. Here, ConvolutionLayer and AffineLayer don't have the flag.
```
from dnnet.config import Config
Config.enable_gpu()

# Do something here.

# AffineLayer uses GPU.
model.add(AffineLayer(output=512, weight_initialization=He()))
# BatchNormLayer uses CPU regardless of Config.enable_gpu().
model.add(BatchNormLayer(force_cpu=True))
```


## Supported Methods
### Layers
* Affine
* Convolution
* Activation
* Pool
* Batch Normalization
* Dropout

### Activation Functions
* Sigmoid
* ReLU
* ELU
* Tanh
* Softmax

### Optimization Methods
* SGD
* Momentum
* AdaGrad
* Adam
* AdaDelta
* RMSProp

### Weight Initialization Methods
* Xavier's method
* He's method
* Default

### Loss Functions
* MultinomialCrossEntropy for multinomial classification.
* BinomialCrossEntropy for binary classification.
* SquaredError for regression.


# Installation
## Requisites
* python 3.4 or later
* numpy 1.12.0 or later
* matplotlib

If you'd like to use GPU, you need to install the follows additionally.
* CUDA (eg. CUDA 10.0)
* CuDNN (eg. CuDNN7.6.5)
* cupy (eg. cupy-cuda100==7.0.0)

## Install dnnet by pip.
```
pip install dnnet
```

## Install dnnet from source.
dnnet doesn't require any complicated path-settings.<br>
You just download scripts from github, place it wherever you like,<br>
and you add some lines like below in your scripts.

```
import sys
sys.path.append('<path-to-dnnet-root-dir>')

from dnnet.neuralnet import NeuralNetwork
```

## Setup environment from scratch (Optional)
In this section, setting up python environment from scratch is described.<br>
"From scratch" means that you're supposed to use brand-new computer,<br>
no python packages (even python itself!) and relevant libraries are installed.

It may also be useful when you start new python project. In this case,<br>
you will partially execute the following steps.

### Setup Python Virtual Environment
#### Assumption
* Use python3
* Make directory for pyenv in "/home/<user-name>/Documents"
* Root directory of your python virtual env is in "/home/<user-name>/Work/py352_ws"
* "/home/<user-name>/Work/py352_ws/" is your working directory

#### Setup procedure
* Install required packages
```
$ sudo apt-get install git gcc make openssl libssl-dev libbz2-dev libreadline-dev libsqlite3-dev zlib1g-dev libffi-dev
```

* Install tkinter(This is required to use matplotlib in virtualenv)
```
$ sudo apt-get install python3-tk python-tk tk-dev
```

* Install pyenv
```
   $ cd ~
   $ git clone git://github.com/yyuu/pyenv.git ./pyenv
   $ mkdir -p ./pyenv/versions ./pyenv/shims
```

* Set paths
Add the following description in ~/.bashrc
```
export PYENV_ROOT=${HOME}/Documents/pyenv
if [ -d "${PYENV_ROOT}" ]; then
  export PATH=${PYENV_ROOT}/bin:$PATH
  eval "$(pyenv init -)"
fi
```

And then execute the follows.
```
   $ exec $SHELL -l
   $ . ~/.bashrc
```

* Install pyenv-virtualenv
```
   $ cd $PYENV_ROOT/plugins
   $ git clone git://github.com/yyuu/pyenv-virtualenv.git
```

* Install python 3.5.2
```
   $ pyenv install 3.5.2
```

* Setup local pyenv
```
   $ mkdir -p ~/Work/py352_ws
   $ pyenv virtualenv 3.5.2 <name of this environment>
```
<name of this environment> can be like py352_env, python3_env, or anything you like.<br>
Here, it's assumed that you named the environment as "py352_env".
```
   $ cd ~/Work/py352_ws
   $ pyenv local py352_env
   $ pip install --upgrade pip
```


# Example
## MNIST
* Run neural network for mnist.
```
cd <path-to-dnnet>/examples/mnist
python mnist.py
```

If you get an error "ImportError: Python is not installed as a framework.",
it might be because of matplotlib issue.(This happened to me when working with MacOS.)

In the case, please try the follow.
```
cd ~/.matplotlib
echo "backend: TkAgg" >> matplotlibrc
```


# Usage in your project
## If you pip installed dnnet
```
from dnnet.neuralnet import NeuralNetwork
```

## If you git cloned dnnet
```
import sys
sys.path.append('<path-to-dnnet-root-dir>')

from dnnet.neuralnet import NeuralNetwork
```

For example, if dnnet directory is in ~/Work/dnnet,
do like below.
```
import os
import sys
sys.path.append(os.path.join(os.getenv('HOME'), 'Work/dnnet'))

from dnnet.neuralnet import NeuralNetwork
```
