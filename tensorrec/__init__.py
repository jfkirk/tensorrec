from .tensorrec import TensorRec
from . import eval
from . import loss_graphs
from . import util

__version__ = '0.1'

__all__ = [TensorRec, eval, util, loss_graphs]

# Suppress TensorFlow logs
import logging
logging.getLogger('tensorflow').disabled = True
