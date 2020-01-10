# Authors: Daichi Yoshikawa <daichi.yoshikawa@gmail.com>
# License: BSD 3 clause

from dnnet.config import Config

import logging
logger = logging.getLogger('dnnet.log')


if Config.use_gpu():
    import cupy as cp
    import numpy as np
    logger.debug('cupy is imported as cp.')
else:
    import numpy as cp
    import numpy as np
    logger.debug('numpy is imported as cp.')
