dnnet
=====
Implementation of Deep Neural Network with numpy.
Less dependencies, easier to use.

# Get started
## Setup Python Virtual Environment
### Assumption
* Use python3
* Make directory for pyenv in "/home/<user-name>/Documents"
* Root directory of your python virtual env is in "/home/<user-name>/Work/py352_ws"
* "/home/<user-name>/Work/py352_ws/" is your working directory

### Setup procedure
* Install required packages
```
$ sudo apt-get install git gcc make openssl libssl-dev libbz2-dev libreadline-dev libsqlite3-dev
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

If you're going to use theano in the resulting env,
```
env PYTHON_CONFIGURE_OPTS=--enable-shared pyenv install 3.5.2
```
else ...
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

## Install requisites
* python 3.4 or later
* numpy 1.12.0 or later
* matplotlib

Go to your virtual environment like /home/<user-name>/Work/py352_ws,
and do the follows.
```
pip install numpy
pip install matplotlib
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

# Use in your scripts (If you pip installed dnnet)
```
from dnnet.neuralnet import NeuralNetwork
```

# Use in your scripts (If you git cloned dnnet)
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