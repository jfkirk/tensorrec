import logging
from .tensorrec import TensorRec
from . import eval
from . import input_utils
from . import loss_graphs
from . import representation_graphs
from . import prediction_graphs
from . import session_management
from . import util

__version__ = '0.25.4'

__all__ = [
    TensorRec, eval, util, loss_graphs,
    representation_graphs, prediction_graphs,
    session_management, input_utils
]

# Suppress TensorFlow logs
logging.getLogger('tensorflow').disabled = True
