"""This module implements DNN Exceptions."""


class DNNetIOError(IOError):
    """Raised when IO failure occurs."""
    def __init__(self, msg):
        super().__init__(msg)


class DNNetRuntimeError(RuntimeError):
    """Raised when runtime error occurs."""
    def __init__(self, msg):
        super().__init__(msg)
