"""Collection of dummy classes for MPI functionality in serial runs."""

import numpy as np


class DummyComm:
    """Dummy replacement for `comm` communicator of `mpi4py`.""" 

    def __init__(self):
        print("Instantiating dummy communicator")

    def allreduce(self, x, mode=None, op=None):
        return x

    def allgather(self, x):
        return np.array([x])

    def reduce(self, x, mode=None, op=None):
        return x

    def bcast(self, x, root=None):
        return x

    def barrier(self):
        return 

    
class DummyMPI:
    """Dummy replacement for `MPI` class of `mpi4py`."""
    
    def __init__(self):
        self.MIN = None
        self.MAX = None
        self.CHAR = None
        self.SHORT = None
        self.INT = None
        self.LONG = None
        self.LONG_LONG = None
        self.FLOAT = None
        self.DOUBLE = None
        self.BOOL = None

        
