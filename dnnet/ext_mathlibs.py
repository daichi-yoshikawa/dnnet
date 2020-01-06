# Authors: Daichi Yoshikawa <daichi.yoshikawa@gmail.com>
# License: BSD 3 clause

from dnnet.config import Config


if Config.use_gpu():
    import cupy as cp
    import numpy as np
else:
    import numpy as cp
    import numpy as np
