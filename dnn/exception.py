"""This module implements DNN Exceptions."""

class DNNIOError(IOError):
    """Raised when IO failure occurs."""
    def __init__(self, msg):
        super().__init__(msg)


class DNNRuntimeError(RuntimeError):
    """Raised when runtime error occurs."""
    def __init__(self, msg):
        super().__init__(msg)
