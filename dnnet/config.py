# Authors: Daichi Yoshikawa <daichi.yoshikawa@gmail.com>
# License: BSD 3 clause

import sys
import importlib
from dnnet.__version__ import __title__


class Config:
    __use_gpu = False

    @classmethod
    def enable_gpu(cls):
        cls.__use_gpu = True
        cls.reload_all_modules()

    @classmethod
    def disable_gpu(cls):
        cls.__use_gpu = False
        cls.reload_all_modules()

    @classmethod
    def reload_all_modules(cls):
        tgt_modules = [m for m in sys.modules if m.startswith(__title__)]
        tgt_modules.pop(tgt_modules.index(__name__))

        for module in tgt_modules:
            importlib.reload(sys.modules[module])

    @classmethod
    def use_gpu(cls):
        return cls.__use_gpu
